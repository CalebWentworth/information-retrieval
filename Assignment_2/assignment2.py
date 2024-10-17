import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
from argparse import ArgumentParser
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

#removes stop words from a given string
def remove_stopwords(text, lst_stopwords):
    words = text.split()
    filtered_sentence = [w for w in words if not w.lower() in lst_stopwords]
    clean_text = ' '.join(filtered_sentence).strip()
    return clean_text

#
def preprocess_answers(answer_dict):
    stop_words = stopwords.words('english')
    for item in answer_dict:
        text = item['Text']
        text = remove_stopwords(text, stop_words)
        text = text.lower()
        text = remove_tags(BeautifulSoup(text, "html.parser"))
        text = re.sub(r'[^\w\s]', '',text)
        
        item['Text'] = text
    return answer_dict

def remove_tags(soup):
    for data in soup:
        if hasattr(data, 'decompose'):
            data.decompose()
    return ' '.join(soup.stripped_strings)

def preprocess_query_remove_html(data):
    for item in data:
        text = item['Text']
        #print(item['Id'])
        text = remove_tags(BeautifulSoup(text, "html.parser"))
        item['Text'] = text
    print("Tags removed")

def tokenize_doc(doc):
    #turn text into list of tokens
    text = doc['Text']
    text = text.split()
    return text

def doc2vec(doc):    
    text = doc['Text']
    text = text.split()
    return {doc['Id'] : text}

#returns doc id : string representation of the question.
def answer_reduce(data):
    answers = {}
    for doc in data:
        doc2vec(doc)
        answers.update(doc2vec(doc))
    return answers

def generate_corpus(answers):
    corpus = []
    #generates the set of all words in the search space
    for item in answers:
        corpus += tokenize_doc(item)
    return set(corpus)

def index_answers(corpus, docs):
    inverted_index = {term: [] for term in corpus}
    
    for doc_id, terms in docs.items():
        for term in terms:
            if term in corpus:
                inverted_index[term].append(doc_id)
            
                
    return inverted_index

def idf(index,term,docs):
    doc_freq = len(index[term])
    return math.log(doc_freq+1/len(docs)+1)

def tf(term, doc):
    freq = doc.count(term)
    doc_len = len(doc)
    return freq / doc_len if doc_len > 0 else 0

def tf_idf(index,term,docs,doc):
    if term in doc:
        return tf(term,doc)*idf(index,term,docs)
    else:
        return 0
    
def avg_len(doc_list):
    sum=0
    for doc in doc_list:
        sum+=len(doc)
    
    return sum

def bm25(term_freq, doc_len, avg_doc_len, idf, k1=1.75, b=0.75):
    return idf * ((term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len))))

def bm_search(query, index, doc_list, idf_values, k1=1.75, b=0.75):
    query_terms = query.split()
    scores = defaultdict(float)
    avg_doc_len = sum(len(doc) for doc in doc_list.values()) / len(doc_list)
    
    for term in query_terms:
        if term in index:
            idf = idf_values[term]
            for doc_id in index[term]:
                doc = doc_list[doc_id]
                term_freq = doc.count(term)
                doc_len = len(doc)
                scores[doc_id] += bm25(term_freq, doc_len, avg_doc_len, idf, k1, b)
    
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def precompute_idf(index, num_docs):
    idf = {}
    for term, doc_ids in index.items():
        df = len(doc_ids)
        idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
    return idf

def tf_search(query, index, doc_list, idf_values):
    query_terms = query.split()
    scores = defaultdict(float)
    
    for doc_id, doc in doc_list.items():
        score = sum(tf(term, doc) * idf_values[term] for term in query_terms if term in index)
        scores[doc_id] = score
    
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def parse_query(query_dict):
    stop_words = stopwords.words('english')
    querys = {}
    for item in query_dict:
        id = item['Id']
        title = item['Title']
        body = item['Body']

        title = remove_stopwords(title,stop_words)
        
        title = title.lower()
        title = remove_tags(BeautifulSoup(title, "html.parser"))
        title = re.sub(r'[^\w\s]', '',title)
        
        body = remove_stopwords(body,stop_words)
        body = body.lower()
        body = remove_tags(BeautifulSoup(body, "html.parser"))
        body = re.sub(r'[^\w\s]', '',body)
        
        text = title+' '+body
        querys[id] = text

    return querys

def process_query(query_id, query_text, index, doc_list, idf_vals):
    bm_search_results = bm_search(query_text, index, doc_list, idf_vals)
    tf_search_results = tf_search(query_text, index, doc_list, idf_vals)
    
    bm_search_results_top5 = bm_search_results[:5]
    tf_search_results_top5 = tf_search_results[:5]
    
    bm_ranked = {doc_id: rank for rank, (doc_id, _) in enumerate(bm_search_results_top5, start=1)}
    tf_ranked = {doc_id: rank for rank, (doc_id, _) in enumerate(tf_search_results_top5, start=1)}
    
    return query_id, bm_ranked, tf_ranked

def run_querys(querys, idf_vals, index, doc_list):
    bm_results = []
    tf_results = []
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_query, query_id, query_text, index, doc_list, idf_vals)
            for query_id, query_text in querys.items()
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Queries"):
            query_id, bm_ranked, tf_ranked = future.result()
            bm_results.append({query_id: bm_ranked})
            tf_results.append({query_id: tf_ranked})
    
    return bm_results, tf_results

def result_gen(results,f_name):
    bm_result = results[0]
    tf_result = results[1]
    print(bm_result)
    q0="Q0"
    run_name = 'Caleb'
    bm_output = f_name.rsplit('.', 1)[0] +'_bm25'+'.tsv'
    tf_output = f_name.rsplit('.', 1)[0] +'_tf'+'.tsv'

    # tsv format: query_id "Q0" answer_id 0 rank run_name
    with open(bm_output, 'w') as f:
        for result in bm_result:
            query_id = list(result.keys())[0]
            for answer_id, rank in result[query_id].items():
                f.write(f"{query_id}\t{q0}\t{answer_id}\t0\t{rank}\t{run_name}\n")

    with open(tf_output, 'w') as f:
        for result in tf_result:
            query_id = list(result.keys())[0]
            for answer_id, rank in result[query_id].items():
                f.write(f"{query_id}\t{q0}\t{answer_id}\t0\t{rank}\t{run_name}\n")


def main():
    nltk.download('stopwords')
    # this section parses program arguments and 
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='search domain file e.g. Answers.json',default="Answers.json")
    parser.add_argument('-q', '--query', required=True, help='query source files e.g. topics_1.json',default="topics_1.json")

    args = parser.parse_args()
    input_file = args.input
    query_file = args.query

    #prepares the search space(ansers)  and queries (querys)
    with open(input_file,'r') as file:
        answers = json.load(file)

    with open(query_file,'r') as file:
        querys = json.load(file)
    #processes the search space by removing stop words, setting words to lower case, removing punctuation
    #and removing html tags
    preprocess_answers(answers)
    # simplifies the format of the answers to doc id : text
    doc_list = answer_reduce(answers)
    # generates the set of all terms in the answer space 
    corpus = generate_corpus(answers)
    #creates an inverted index of the corpus
    index = index_answers(corpus,doc_list)
    #creates a list of idf values of each term in the corpus
    idf_vals = precompute_idf(index,len(doc_list))
    #parses the queries in the same way as the answers
    q_input = parse_query(querys)
    # runs the parsed queries through the search engine
    result = run_querys(q_input,idf_vals,index,doc_list)
    #produces TREC result files
    result_gen(result,query_file)