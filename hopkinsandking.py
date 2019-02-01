# Class method to implement Hopkins and King's (2010) method

import re # regular expressions
import string

import numpy as np
from numpy.random import choice
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

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
    return np.linalg.inv(Pmat.T.dot(Pmat)).dot(Pmat.T)



class HopkinsKingCategoryCount(BaseEstimator, TransformerMixin):
    """
    This implements Hopkins-King, ''A Method of Automated Nonparametric Content
    Analysis for Social Science''.

    Parameters
    ----------
    setOk : array-like
        This comes, for example, from applying np.nonzero(A==condition) to a matrix
    
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
        
    lamb : int (default=0)
        This is the regularization parameter, which should be non-negative since
        it is a minimization procedure.
        
    epsilon : float (default=0.001)
        Desired accuracy of the estimation.
    
    max_iters : integer (default=100)
    
    doprint : boolean, optional (default=False)
        Prints different results of the estimation steps.
        
    printbatch : integer, optional if doprint==True (default=10)
        After how many iterations to print the loss function.
        
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    

    Attributes
    ----------
    Lest_ : array-like, shape (N, T)
        This is the estimate of L.
    
    loss_ : float
        The root-square of the mean square error of the observed elements.
        
    iters_ : int
        Number of iterations it took the algorithm to get the desired
        precision given by the parameter 'epsilon'.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1,2,np.nan, 0],[np.nan,2,2,1],[3,0,1,3],[0,0,2,np.nan]])
    
    Notes
    -----
    
    
    """
    
   

    def __init__(self, kwords=25, min_freq=10, 
                 random_seed = None,
                 copy=True):
        self.kwords = kwords 
        self.min_freq = min_freq
        self.random_seed = random_seed
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
        
        # I should check everthing is OK
        assert ((self.kwords > 0) & (self.kwords < len(X.shape[0]))), "0 < kwords < np.log2(length of array)"
        assert (len(X.shape) == 1), "The input X for the fit should be an array of text (long descriptions)"
        assert (len(y.shape) == 1), "The input y for the fit should be an array of text (categories)"
        
        # Create the cleaning function, clean the text, and create bag of words with the frequencies
        ct = CleanText()
        str_clean, word_count = ct.fit_transform(X.values)
        
        # Get a small k-sample of words
        
        
        
        words2analyze = getWords2Analyze(word_count, words2keep=None, min_freq=10)
        
        
        # Removing the most and least common words
        #binary_2drop = ((word_count['frequency']==word_count['frequency'].max()) | 
        #        (word_count['frequency']<=self.min_freq))
        #words2drop = word_count.loc[binary_2drop]
        #words2keep = word_count.loc[~binary_2drop]
        
        #words2keep_as_set = set(words2keep.word.values)
        
        # Keep only the important words
        words2analyze = getWords2Analyze(word_count, words2keep=None, min_freq=10)
        
        # Restricting the analysis to a subset of words
        if(self.random_seed is not None): np.random.seed(self.random_seed)
        wordsconsidered = choice(words2keep.word.values, 
                                 size=self.kwords, 
                                 p = words2keep.frequency.values/sum(words2keep.frequency.values), 
                                 replace=False)
        
        # Convert the bag of words into an array
        vectext = words2analyze.apply(lambda x: " ".join(list(set(x.split()).intersection(set(wordsconsidered)))))
        
        # From the array, create a Data Frame
        Xdf = create_Xmat(vectext)
        Xdf['category'] = y.values
        Xdf['count'] = 1.0
        
        # 10- 
        Fsd = Xdf[['wordcombination', 'category', 'count']].groupby(by = ['wordcombination', 'category'], as_index=False).sum()
        Fsd['tot_by_category'] = Fsd.groupby(by = ['category'])['count'].transform('sum')
        Fsd['P_s_given_D'] = Fsd['count']/Fsd['tot_by_category']
        
        # 11-
        Pmat_SgD = Fsd.pivot(index = 'wordcombination', columns='category', values = 'P_s_given_D').fillna(0.0)
        
        self.Pmat_SgD_ = Pmat_SgD
        self.Qtransform_ = (np.linalg.inv((final_Pmat_SgD.T).dot(final_Pmat_SgD))).dot(final_Pmat_SgD.T)
        
        # Return the transformer
        return self

    def transform(self, X):
        """ 
        Actually returning the estimated matrix, in which we have 
        imputed the missing values of X (matrix Y).
        
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (N, T)
            The input data to complete.
            
        Returns
        -------
        X_transformed : array, shape (N, T)
            The array the completed matrix.
        """
        try:
            getattr(self, "Qtransform_")
        except AttributeError:
            raise RuntimeError("You must estimate the model before transforming the data!")
                
        
        # create the array of words to analyze
        words2analyze = d
        
        # 1 - 
        vectext_test = words2analyze.apply(lambda x: " ".join(list(set(x.split()).intersection(set(wordsconsidered)))))

        Xdf_test = create_Xmat(vectext_test)
        Xdf_test['category'] = longdf_test['cpc_1_text'].values

        Xdf_test['count'] = 1.0
        
        Ps = Xdf_test.groupby(by = ['wordcombination'])['count'].sum()
        
        # 2 - Completing the word combinations
        missingwordcombinations = set(Ps.index).difference(set(self.Pmat_SgD_.index))

        missing_Pmat_SgD = pd.DataFrame(np.zeros((len(missingwordcombinations), self.Pmat_SgD_.shape[1])),
                               index = list(missingwordcombinations), columns = self.Pmat_SgD_.columns)
        
        
        return self.Lest_

