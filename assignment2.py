import nltk, json
from nltk.corpus import stopwords
from bs4 import BeautifulSoup



def initialize():
    nltk.download('stopwords')
    

def remove_stopwords(text, lst_stopwords):
    words = text.split()
    filtered_sentence = [w for w in words if not w.lower() in lst_stopwords]
    clean_text = ' '.join(filtered_sentence).strip()
    return clean_text

def preprocess_questions(question_dictionary):
    stop_words = stopwords.words('english')
    for item in data:
        title = item['Title']
        title = remove_stopwords(title, stop_words)
        body = item['Body']
        body = remove_stopwords(body, stop_words)
        item['Title'] = title
        item['Body'] = body
    return question_dictionary

def remove_tags(soup):
    for data in soup:
        data.decompose
    return ' '.join(soup.stripped_strings)

def preprocess_remove_html(question_dictionary):
    for item in data:
        body = item['Body']
        body = remove_tags(BeautifulSoup(body, "html.parser"))
        item['Body'] = body
    
with open('topics_1.json','r') as file:
    data = json.load(file)

