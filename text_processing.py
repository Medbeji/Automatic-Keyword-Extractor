from singleton import Singleton
import spacy
from tqdm import tqdm 
import multiprocessing as mp
import time
from pprint import pprint


class TextLemmatizer(metaclass=Singleton):

    def __init__(self, language='en'):
        """ Initiate a TextProcessor by passing the step of normalization """
        model_name = "en_core_web_md" if language == 'en' else 'fr_core_news_md'
        self.nlp = spacy.load(model_name)
        self.n_core = mp.cpu_count()
        
    def process(self, texts=None):
        """ Apply normalization on selected data in params """
        batch_size = 500 if len(texts) > 1000 else 250
        result = []
        for doc in tqdm(self.nlp.pipe(texts, disable=["ner", "parser"], n_threads=self.n_core, batch_size=batch_size)):
            new_text = ''
            for token in doc:
                if token.pos_ in ['PROPN', 'NOUN', 'ADJ'] and not token.is_stop and not token.like_num:
                    new_text += ' ' + token.lemma_
            result.append(new_text)
        return result


if __name__ == "__main__":
    print('Building first lemmatizer :')
    start = time.time()
    tp = TextLemmatizer(language='en')
    end = time.time()
    print(f'Ended after {end-start:2f}s')

    data = ['Hello world this is a hydrogen vehicle', 'this is a vehicle', 'hydrogen', 'baby','I know that trendency will be a hydrogen vehicle']
    pprint(tp.process(data))
    
