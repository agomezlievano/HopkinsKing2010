# HopkinsKing2010
Here we implement, and create a python class, for Hopkins and King's (2010) method. The paper is title "A Method of Automated Nonparametric Content Analysis for Social Science", and I highly recommend it. As many of King's papers, it is extremely clear and simple, yet very insightful.

The main point of the paper is that for many purposes and research questions in the social (and economic) sciences, one is interested in the aggregates. That is, one is typically not so much interested in what Alice and Bob are doing as individuals, but in what Alices and Bobs do as groups of people. The group-level variables are of course the result of aggregating individual-level variables.

For example, I am interested in quantifying the number of people employed in different occupations in a given city. In principle, I should just take the information of all individuals, categorize each individual into one of different occupational categories, and then aggregate to the level of the city. What is the problem that Hopkins and King are trying to solve? The problem is that very often individual observations are not assigned to any category. Thus, this is yet another instance of "missing value" imputation, of which I've become very fond of because it is directly related to questions of causal inference.

Hopkins and King's work is in the context of categorizing individual documents into different topics, and their interest is in quantifying the proportion of documents per topic in a corpus of text. 

Their insight is that one can skip entirely the step of "assigning a category to each document", before "aggregating to create the counts". 
