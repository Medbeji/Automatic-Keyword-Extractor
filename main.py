from rake import Rake
from textrank import TextRank
from tf_idf import TFIDF


def main():
    docs = ["the house had a tiny little mouse",
            "the cat saw the mouse",
            "the mouse ran away from the house",
            "the cat finally ate the mouse",
            "the end of the mouse story"
            ]
    # TF-IDF Tester 
    tfidf_handler = TFIDF(use_idf=True, ngram=(1, 3))
    tfidf_handler.analyse_corpus(document=docs)
    df = tfidf_handler.extract_keywords(doc_idx=0)
    print(df.head())


if __name__ == "__main__":
    # execute only if run as a script
    main()
