from flask import Flask , request, jsonify
from text_processing import TextLemmatizer
from bow import Bow

app = Flask(__name__)


@app.route('/thesaurus', methods=['POST'])
def get_keywords():
    """
    Extract keywords from selected corpus 
    """
    # getting the json object from POST body
    json_object = request.get_json()

    # check for body format
    if 'texts' not in json_object.keys() or len(json_object['texts']) == 0:
        return {
            'Please check the body of your requests, no texts are found!'
        }, 200
        
    texts = json_object['texts']

    # check if using lemmatizer
    if json_object['use_lemmatizer'].lower() == 'true':
        # Initialize the lemmatizer 
        lemmatizer = TextLemmatizer(language=json_object['language'] if 'language' in json_object.keys() else 'en')
        texts = lemmatizer.process(texts)
        
    bow = Bow() 
    return jsonify(bow.process(texts=texts))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


