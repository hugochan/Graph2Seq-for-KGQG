import re
import string
import json
from nltk.tokenize import wordpunct_tokenize#, word_tokenize


tokenize = lambda s: wordpunct_tokenize(s)


ctx_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
def get_text_overlap(query, entity_name):
    '''Return meaningful overlapped sub-string between two strings'''
    def longest_common_substring(s1, s2):
       m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
       longest, x_longest = 0, 0
       for x in range(1, 1 + len(s1)):
           for y in range(1, 1 + len(s2)):
               if s1[x - 1] == s2[y - 1]:
                   m[x][y] = m[x - 1][y - 1] + 1
                   if m[x][y] > longest:
                       longest = m[x][y]
                       x_longest = x
               else:
                   m[x][y] = 0
       return s1[x_longest - longest: x_longest]

    sub_seq = longest_common_substring(tokenize(query.lower()), tokenize(entity_name.lower()))
    if len(set(sub_seq) - ctx_stopwords) == 0:
        return ''
    else:
        return sub_seq


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[%s]' % re.escape(string.punctuation))
def normalize_answer(s):
    """Lower text and remove extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_ndjson(file, return_type='array'):
    if return_type == 'array':
        return load_ndjson_to_array(file)
    elif return_type == 'dict':
        return load_ndjson_to_dict(file)
    else:
        raise RuntimeError('Unknown return_type: %s' % return_type)

def load_ndjson_to_array(file):
    data = []
    try:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def load_ndjson_to_dict(file):
    data = {}
    try:
        with open(file, 'r') as f:
            for line in f:
                data.update(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e
