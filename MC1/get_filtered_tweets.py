import string

import operator
from collections import defaultdict
import warnings

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer

from itertools import chain

import preprocessor as p

import pandas as pd

from gensim.models import Word2Vec
import json
import numpy
from collections import Counter

# Ignore warnings
warnings.filterwarnings('ignore')

p.set_options(p.OPT.EMOJI,p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.SMILEY)
stemmer = SnowballStemmer("english")

# file_tweets = './static/Microblogs_processed.txt'
# file_w2v = './static/w2v.model'
# file_w2v_index = './static/word_tweetIndices.txt'

file_tweets = 'HW7/static/Microblogs_processed.txt'
file_w2v = 'HW7/static/w2v.model'
file_w2v_index = 'HW7/static/word_tweetIndices.txt'

# Read original data
def get_original_data():
    data1 = pd.read_csv("Microblogs.csv", error_bad_lines=False, encoding="ISO-8859-1")
    data1 = data1.dropna()
    data1['ID'] = data1['ID'].astype(int)
    data1['text'] = data1['text'].astype(str)
    data1['Created_at'] = pd.to_datetime(data1['Created_at'], errors='coerce', infer_datetime_format=True)
    data1['Location'] = data1['Location'].astype(str)
    data1 = data1.dropna(subset=['Created_at'])
    print(data1.shape)
    return data1

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


#Get preprocessed Data - with stemmed words
def get_preprocessed_data():
    data1 = pd.read_csv(file_tweets, error_bad_lines=False, delimiter="\t", encoding="ISO-8859-1", index_col=False)
    data1 = data1.dropna()
    data1['ID'] = data1['ID'].astype(int)
    data1['text'] = data1['text'].astype(str)
    data1['Created_at'] = pd.to_datetime(data1['Created_at'], errors='coerce', infer_datetime_format=True)
    data1['Words'] = data1['Words'].astype(str)
    data1['Lat'] = data1['Lat'].astype(float)
    data1['Long'] = data1['Long'].astype(float)
    data1 = data1.dropna(subset=['Created_at'])
    print(data1.shape)
    return data1

#Get preprocessed Data - with stemmed words
def get_17_data():
    data1 = pd.read_csv("tweets_on_17.txt", error_bad_lines=False, delimiter="\t", encoding="ISO-8859-1", index_col=False)
    data1 = data1.dropna()
    data1['ID'] = data1['ID'].astype(int)
    data1['text'] = data1['text'].astype(str)
    data1['Created_at'] = pd.to_datetime(data1['Created_at'], errors='coerce', infer_datetime_format=True)
    data1['Words'] = data1['Words'].astype(str)
    data1['Lat'] = data1['Lat'].astype(float)
    data1['Long'] = data1['Long'].astype(float)
    data1 = data1.dropna(subset=['Created_at'])
    print(data1.shape)
    return data1

#perform preprocessing on the tweet
table = str.maketrans({key: None for key in string.punctuation})
def process_tweet(tweet):
    tweet = p.clean(tweet)
    stop = stopwords.words('english') + list(string.punctuation)
    tweet = tweet.translate(table)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    words = [stemmer.stem(word) for word in words]
    return words

#this is used by the word2vec to read from the file
class TweetSentences(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in open(self.fname, 'r'):
            line = line.strip()
            yield line.split(',')

#function to preprocess each tweet and store list of words corresponding to tweet
def process_tweet_dataset():
    dataset = get_original_data()
    count = 0
    file = open('Microblogs_processed.txt', 'w', encoding="utf-8")
    file1 = open('sentences.txt', 'w', encoding="utf-8")
    file2 = open('tweets_on_17.txt', 'w', encoding="utf-8")
    file.write('ID\tCreated_at\tLat\tLong\ttext\tWords\n')
    file2.write('ID\tCreated_at\tLat\tLong\ttext\tWords\n')
    for index, row in dataset.iterrows():
        tweet = row['text']
        if not tweet:
            continue
        sentence = process_tweet(tweet)
        count = count + 1
        print(count)
        if (len(sentence) == 0):
            continue
        x = ','.join(sentence)
        line = str(row['ID']) + "\t" + str(row['Created_at']) + "\t" + str(row['Location']).split()[0] + "\t" + \
               str(row['Location']).split()[1] + "\t" + row['text'].replace('\t', ' ') + "\t" + x + "\t\n"
        file.write(line)
        if str(row["Created_at"].date()) == "2011-05-17":
            file2.write(line)
        file1.write(x + "   \n")
    file.close()
    file1.close()

def get_word_tweet_indices():
    data = get_preprocessed_data()
    print(data)
    word_indices = defaultdict(list)
    for record in data.itertuples():
        words = record.Words.split(",")
        for word in words:
            word_indices[word].append(record.Index)
    with open('word_tweetIndices.txt', 'w') as fp:
        json.dump(word_indices, fp, cls=MyEncoder)

def get_tweets_from_word(symptoms, tweet_dataset):
    catch_words = [x.strip() for x in symptoms.split(',')]
    catch_words = [stemmer.stem(word) for word in catch_words]
    relevantWords = defaultdict(int)
    for catch_word in catch_words:
        synonyms = wordnet.synsets(catch_word)
        lemmas = list(set(chain.from_iterable([catch_word.lemma_names() for catch_word in synonyms])))
        for word in lemmas:
            if word not in relevantWords.keys():
                relevantWords[word] = 0
    relevant_tweet = pd.DataFrame(columns=["text",'Lat',"Long", "Created_at"])
    relevant_tweets_only = []
    not_relevant_tweet = []
    i = 0
    for index, row in tweet_dataset.iterrows():
        i = i+1
        print(i)
        items = row['Words']
        items_tweet = [x.strip() for x in items.split(',')]
        found = 0
        for word in items_tweet:
            if word in relevantWords.keys():
                relevantWords[word] += 1
                # relevant_tweets_only.append(row["text"])
                relevant_tweet.loc[len(relevant_tweet)] = [row["text"], row["Lat"], row["Long"],row["Created_at"] ]
                found  = 1
                break
        # if found == 0:
        #     not_relevant_tweet.append(original_tweet)
    # Remove words with zero occurrences
    # relevantWords = {k: v for k, v in relevantWords.items() if v != 0 and k != catch_word}
    return relevantWords, relevant_tweet


def get_tweet_word2vec(symptoms, tweet_dataset):
    model = Word2Vec.load('w2v.model')
    catch_words = [x.strip() for x in symptoms.split(',')]
    catch_words = [stemmer.stem(word) for word in catch_words]
    relevantWords = defaultdict(int)
    for word in catch_words:
        similarWords = model.most_similar(word, topn=20)
        print(similarWords)
        if word not in relevantWords.keys():
            relevantWords[word] = 1
        for t in similarWords:
            if t[1] > 0.50 and t[0] not in relevantWords.keys():
                relevantWords[t[0]] = t[1]
    # catch_words = [stemmer.stem(word) for word in relevantWords.keys()]
    relevant_tweet = pd.DataFrame(columns=["text", 'Lat', "Long", "Created_at"])
    not_relevant_tweet = pd.DataFrame(columns=["text", 'Lat', "Long", "Created_at"])
    for index, row in tweet_dataset.iterrows():
        items = row['Words']
        items_tweet = [x.strip() for x in items.split(',')]
        found  =0
        for word in items_tweet:
            if word in relevantWords.keys():
                relevant_tweet.loc[len(relevant_tweet)] = [row["text"], row["Lat"], row["Long"], row["Created_at"]]
                found = 1
                break
    #     if found == 0:
    #         not_relevant_tweet.loc[len(not_relevant_tweet)] = [row["text"], row["Lat"], row["Long"], row["Created_at"]]
    # not_relevant_tweet.to_csv("not_flu_tweets.csv", encoding='utf-8')
    # Remove words with zero occurrences
    # relevantWords = {k: v forz k, v in relevantWords.items() if v != 0}
    return relevantWords, relevant_tweet

def word_2_vec_computation():
    sentences = TweetSentences('sentences.txt')  # a memory-friendly iterator
    model = Word2Vec(sentences, iter=10)
    model.save('w2v.model')

def update_word2vec_model(tweet_subset):
    print(len(tweet_subset))
    model = Word2Vec(tweet_subset, workers = 4)
    return model

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

def get_events(tweet_dataset):
    tokenized_sentences = []
    # for index, row in tweet_dataset.iterrows():
    #     tweet = row['text'].split()
    #     if not tweet:
    #         continue
    #     tokenized_sentences.append(tweet)
    # sentences = nltk.sent_tokenize(sample)
    # tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = tweet_dataset.text.str.split()
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    # chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    nouns_count = defaultdict(int)
    for sentence in tagged_sentences:
        for word_tag in sentence:
            if word_tag[1] == "NN" or word_tag[1] == "NNS" or word_tag[1] == "NNP" or word_tag[1] == "NNPS":
                w = stemmer.stem(word_tag[0])
                nouns_count[w] += 1
    top_nouns = dict(Counter(nouns_count).most_common(50))
    return top_nouns


#-----------------function which were used for preprocessing of data--------------
# process_tweet_dataset()
# word_2_vec_computation()
# get_word_tweet_indices()
# use Word2vec
# tweet_dataset =  get_preprocessed_data()
# default_search_1 = "flu,sick,fever,aches,pains,fatigue,coughing,vomiting,diarrhea"
# relevantWords, relevant_tweet = get_tweet_word2vec(default_search_1, tweet_dataset)
# relevant_tweet.to_csv("flu_tweets.csv", encoding='utf-8')
# use synset
# tweet_dataset =  get_preprocessed_data()
# default_search_1 = "flu,sick,fever,aches,pains,fatigue,coughing,vomiting,diarrhea"
# relevantWords, relevant_tweet = get_tweets_from_word(default_search_1, tweet_dataset)
# relevant_tweet.to_csv("flu_tweets_synset.csv", encoding='utf-8')

# # Get events
# tweet_dataset = get_17_data()
# get_events(tweet_dataset)
#--------------------------------------------------------------------------
