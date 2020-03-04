from rake import Rake
from textrank import TextRank
from tf_idf import TFIDF

def main():
    _extractors = [Rake(), TextRank(), TFIDF()]
    for extractor in _extractors:
        extractor.extract_keywords()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()