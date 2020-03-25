from rake import Rake
from textrank import TextRank
from tf_idf import TFIDF
import os
from tqdm import tqdm


def split_to_stories(full_story_list):
    stories = []
    index = 0
    for i, story in tqdm(enumerate(full_story_list)):
        if story == '\n':
            stories.append(' '.join(full_story_list[index:i]))
            index = i + 1
    stories.append(''.join(full_story_list[index:]))
    return stories


def load_stories(path='data/'):
    """
    load stories from dailymail
    :path: the path of data
    """
    content = []
    for _ in tqdm(os.listdir(path)):
        if '.story' in _:
            filename = path + _
            with open(filename, 'r', encoding='utf-8') as story_file:
                content.append(split_to_stories(story_file.readlines()[0:10]))
    return content


def main():
    content = load_stories()
    # TF-IDF Tester
    tfidf_handler = TFIDF(use_idf=True, ngram=(1, 3))
    tfidf_handler.analyse_corpus(document=content[0])
    df = tfidf_handler.extract_keywords(doc_idx=0)
    print(df.sort_values(by='tfidf', ascending=False))


if __name__ == "__main__":
    # execute only if run as a script
    main()
