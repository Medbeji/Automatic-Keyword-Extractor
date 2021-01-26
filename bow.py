# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from pprint import pprint
from singleton import Singleton

class Bow(metaclass=Singleton):
    
    """
     This is a classic bag of word keywords extraction method using
     some cleanup and 
    """

    def __init__(self, ngram_range=(1, 3), stop_words='english'):
        pprint('Start Initializing of a bow object')
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range, stop_words=stop_words)
        pprint('End initialization of a bow object')

    def process(self, texts=[], topn=10, ngram=(1, 2, 3)):
        """ process the corpus data and calculate the word count using
         CountVectorizer """

        # train the vectorizer model on the corpus and calcule the number of Vectorizer
        cv_fit = self.vectorizer.fit_transform(texts)
        word_list = self.vectorizer.get_feature_names()
        count_list = cv_fit.toarray().sum(axis=0)
        result_dict = dict(zip(word_list, count_list))

        # transform the calculated data into a dataframe
        df = pd.DataFrame.from_dict(
            result_dict, orient='index', columns=['count'])
        df.sort_values(by='count', ascending=False, inplace=True)

        # splitting the words to unigram, bigram and trigram.
        df.reset_index(inplace=True)
        df['nb_words'] = df['index'].apply(lambda x: len(x.split()))
        dict_structure = {
            '1': 'unigram',
            '2': 'bigram',
            '3': 'trigram'
        }
        result = {}
        for n in list(ngram): 
            subset = df[(df['nb_words'] == int(n))].head(topn)
            result[dict_structure[str(n)]] = subset[['index','count']].to_dict('records')

        return result

if __name__ == "__main__":
    bow_object = Bow()
    data = ['Hello world this is a hydrogen vehicle', 'this is a vehicle', 'hydrogen', 'baby','I know that trendency will be a hydrogen vehicle']
    bow_object.process(texts=data)    
