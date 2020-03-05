from singleton import Singleton
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class TFIDF():

    def __init__(self, use_idf=True, ngram=(1, 1)):
        self._tfidf_vectorizer = TfidfVectorizer(use_idf=use_idf, ngram_range=ngram)
        self.vectors = None

    def analyse_corpus(self, document):
        """ Analyzing corpus and generate tf-idf vectors """
        if self._tfidf_vectorizer is not None:
            # generate count + idf vectorizer at one call.
            self.vectors = self._tfidf_vectorizer.fit_transform(document)
        else:
            # generate exceptions to be handled
            raise Exception('No vectorizer found!')

    def extract_keywords(self, doc_idx=0):
        print('extracting keywords using tf-idf')

        # Get the sparse matrix for the selected document
        try: selected_vectorizer = self.vectors[doc_idx]
        except Exception: return

        # place tf-idf values in a pandas data frame
        df = pd.DataFrame(selected_vectorizer.T.todense(
        ), index=self._tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
        return df.sort_values(by=["tfidf"], ascending=False)
