# Class method to implement Hopkins and King's (2010) method

import re # regular expressions
import string

import numpy as np
from numpy.random import choice
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error

from nltk import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import SnowballStemmer 
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import EnglishStemmer

from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag

from collections import Counter

#import nltk.corpus.stopwords.words as nltkwords
sr = stopwords.words('english')



class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
        
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    
    def keep_alphanumeric(self, input_text):
        return re.sub('[^A-Za-z0-9 ]+', '', input_text)
    
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def unique_words(self, input_text):
        unique_words = list(set(input_text.split()))
        return " ".join(unique_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        # Clean text
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.remove_punctuation).apply(self.keep_alphanumeric).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming).apply(self.unique_words)
        
        return np.array(clean_X)

def create_wordCount(clean_X):
    # Create dictionary of counts per word
    alltext_counter_results = Counter(" ".join(clean_X).split())

    # Create a data frame
    word_count_df = pd.DataFrame(alltext_counter_results.most_common(), columns = ['word', 'frequency'])

    return word_count_df


def sample_kwords(word_count_df, kwords=15, min_freq=10, sample_with_weights=False, p=None, random_seed=None):
    """
    The output of this function is a small array of k words.

    This function assumes that the following code, or something 
    similar, has been run before.

    >>> # Create the cleaning text object
    >>> ct = CleanText()
    >>> text_cleaned, word_count_df = ct.fit_transform(array_of_text)
    """
    # Removing the most and least common words
    binary_2drop = ((word_count_df['frequency']==word_count_df['frequency'].max()) | 
            (word_count_df['frequency']<=min_freq))
    words2keep_df = word_count_df.loc[~binary_2drop]

    if(random_seed is not None): 
        np.random.seed(random_seed)
    if(sample_with_weights is not None):
        p = words2keep_df.frequency.values/sum(words2keep_df.frequency.values)
    wordsconsidered = choice(words2keep_df.word.values,
                                size=kwords, 
                                p=p, 
                                replace=False)
    
    return wordsconsidered


def getWords2Analyze(text_cleaned, kwords_vec):
    """
    The output of this function is an array of same size as text_cleaned,
    but with only the words present in kwords_vec.
    """
    kwords_vec_as_set = set(kwords_vec)


    # Keep only the important words
    words2analyze = text_cleaned.apply(lambda x: " ".join(list(set(x.split()).intersection(kwords_vec_as_set))))

    return words2analyze



def create_combination_str(row):
    return ''.join(map(str, row))

def create_Xmat(vectext, y):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(vectext)
    Xdf = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
    Xdf['wordcombination'] = Xdf.apply(create_combination_str, axis=1)
    Xdf['category'] = y
    Xdf['count'] = 1.0

    return Xdf

def create_Pmat(edgelist_df, columns_names = ['category_destination', 'category_origin', 'count']):
    """
    For documents, the columns_names should be = ['wordcombination', 'category', 'count']
    """
    Fsd = edgelist_df[columns_names].groupby(by = columns_names[:2], as_index=False).sum()

    # The following is more efficient than converting to matrix and doing matrix algebra
    Fsd['tot_by_origin'] = Fsd.groupby(by = [columns_names[1]])[columns_names[2]].transform('sum')
    Fsd['P_dest_given_orig'] = Fsd[columns_names[2]]/Fsd['tot_by_origin']

    Pmat_WgC = Fsd.pivot(index = columns_names[0], columns=columns_names[1], values = 'P_dest_given_orig').fillna(0.0)

    return Pmat_WgC

def getQmat(Pmat):
    Qmat = pd.DataFrame(np.linalg.inv(Pmat.T.dot(Pmat)).dot(Pmat.T),
                        index = Pmat.columns.values, columns = Pmat.index.values)
    return Qmat
    

def create_Pw(vectext):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(vectext)
    Xdf = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
    Xdf['wordcombination'] = Xdf.apply(create_combination_str, axis=1)
    Xdf['count'] = 1.0
    Pw = Xdf.groupby(by = ['wordcombination'])['count'].sum()

    return Pw




class HopkinsKingCategoryCount(BaseEstimator, RegressorMixin):
    """
    This implements Hopkins-King, ''A Method of Automated Nonparametric Content
    Analysis for Social Science''.

    Parameters
    ----------        
    kwords : int (default=25)
        Each document must be characterized by kwords.
        
    random_seed : int (default=None)
        A random seed for sampling the kwords.
    
    min_freq : integer (default=10)
        In the whole training corpus, very rare words will be dropped if they
        appear less than min_freq times.
    
    sample_with_weights : boolean, optional (default=True)
        Sample the kwords according to their frequency in the corpus.
        
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    

    Attributes
    ----------
    selectedkwords_vec_ : array-like
        This is an array of kword selected from the corpus.
    
    PWgC_ : float
        This is the fitted matrix of probabilities mapping word-combinations
        to categories, P(W|C).
                
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1,2,np.nan, 0],[np.nan,2,2,1],[3,0,1,3],[0,0,2,np.nan]])
    
    Notes
    -----
    
    
    """
    
   

    def __init__(self, kwords=25, min_freq=10, 
                 random_seed = None,
                 sample_with_weights = True,
                 copy=True):
        self.kwords = kwords 
        self.min_freq = min_freq
        self.random_seed = random_seed
        self.sample_with_weights = sample_with_weights
        self.copy = copy

    def fit(self, X, y):
        """
        Estimating the matrix P(S|D) which computes the fraction (or
        probability) that a combination of words S was the outcome of
        a document in category D. We compute this by simply counting
        the number of documents within D that have combination of words
        equal to the combinations.
        
        
        Parameters
        ----------
        X : {array-like}, string shape (n_samples)
            The training input that is a vector of raw text. It needs to get cleaned,
            tokenized, etc.
        y : {array-like}, string shape (n_samples)
            Here we have the topics or categories of each text.
            
        Returns
        -------
        self : object
            Returns self.
        """
        
        # Check everthing is OK
        assert (self.kwords > 0), "0 < kwords"
        assert (len(X.shape) == 1), "The input X for the fit should be an array of text (long descriptions)"
        assert (len(y.shape) == 1), "The input y for the fit should be an array of text (categories)"
        
        # Create the cleaning function and clean text
        ct = CleanText()
        text_cleaned = ct.fit_transform(X)

        # Create word count dataframe
        word_count_df = create_wordCount(text_cleaned)

        # Get a small word representation of each document
        self.selectedkwords_vec_ = sample_kwords(word_count_df, kwords=self.kwords, 
                                            min_freq=self.min_freq, 
                                            sample_with_weights=self.sample_with_weights, 
                                            random_seed=self.random_seed)

        # Represent each observation as a vector of presence/absence of the works in the small sample
        words2analyze = getWords2Analyze(pd.Series(text_cleaned), self.selectedkwords_vec_)

        # Create matrix of counts by word combination and label
        Xmat_df = create_Xmat(words2analyze, y)

        # Create P(W|C)
        self.PWgC_ = create_Pmat(Xmat_df, ['wordcombination', 'category','count'])

        # Return the transformer
        return self



    def predict(self, X, y=None):
        """ 
        We use the matrix P(W|C) to predict the probability/frequency vector P(C).
        
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (word combinations)
            The input data to complete.
            
        Returns
        -------
        N_per_C : array, shape (num_categories)
            The array with the estimated fraction/counts of documents per category.
        """
        try:
            getattr(self, "PWgC_")
        except AttributeError:
            raise RuntimeError("You must fit the model before predicting!")
                
        # Clean the test text
        ct = CleanText()
        text_cleaned_test = ct.fit_transform(X)

        # Represent each observation in the test text as a vector of presence/absence of the works in the small sample
        words2analyze_test = getWords2Analyze(pd.Series(text_cleaned_test), self.selectedkwords_vec_) 

        # Create the number of documents in destination category (i.e., documents per word-combination)
        N_per_W = create_Pw(words2analyze_test) 

        # Check whether all wordcombinations in test set are in the training set
        missing_in_train = np.array(list(set(N_per_W.index).difference(set(self.PWgC_.index))))
        if len(missing_in_train)>0: 
            print("There are {} word combinations in test missing in training set.".format(len(missing_in_train)))
            print("That is, {0:.0f} documents in test.".format(np.sum(N_per_W.loc[missing_in_train])))
            print("These represent a fraction of {number:.{digits}f} documents in test.".format(number=np.sum(N_per_W.loc[missing_in_train])/N_per_W.sum(),
                                                                                            digits = 2))
        
        # Expand so that train and test have the same word combinations
        missing_PWgC = pd.DataFrame(np.zeros((len(missing_in_train), self.PWgC_.shape[1])),
                                    index = list(missing_in_train), columns = self.PWgC_.columns)
        final_PWgC = pd.concat((self.PWgC_, missing_PWgC))
        final_N_per_W = N_per_W.reindex(final_PWgC.index, fill_value=0.0)

        # Create the transformation matrix Q
        Qmat = getQmat(final_PWgC)

        # Create category count
        N_per_C = Qmat.dot(final_N_per_W)
        
        return N_per_C

    def loss(self, X, y):
        """
        Computes the root mean squared error.
        """

        # Check everything is ok
        try:
            getattr(self, "PWgC_")
        except AttributeError:
            raise RuntimeError("You must fit the model before scoring predictions!")
        
        assert (len(y.shape) == 1), "The input y for the fit should be an array of text (categories)"

        # Predicted counts
        y_pred = self.predict(X)

        # True counts
        y_real = pd.DataFrame(list(zip(*[y, np.ones_like(y)])), columns=['category', 'count']).groupby('category')['count'].sum()

        # return RMSE
        return np.sqrt(mean_squared_error(y_real, y_pred))




