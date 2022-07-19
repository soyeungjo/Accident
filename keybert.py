from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2

from sentence_transformers import SentenceTransformer

from utils import (
    load_data,
    cleaning,
    mmr,
    josa_delete
)


data_tmp = load_data()
data_tmp['clean_doc'] = data_tmp['doc'].map(lambda x: cleaning(x))


top_n = 3
n_gram_range = (1,)


model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

a = data_tmp.clean_doc[0]

count = CountVectorizer(ngram_range = n_gram_range).fit([a])
candidates = count.get_feature_names_out()

doc_embedding = model.encode([a])
word_embedding = model.encode(candidates)

distances = cosine_similarity(doc_embedding, word_embedding)
tmp_keywords = mmr(doc_embedding, word_embedding, candidates, top_n=top_n, diversity=0.3)


#%%
noun_extractor = LRNounExtractor(verbose=False)
nouns = noun_extractor.train_extract(data_tmp.clean_doc)

candi_words = {} 
    
for word, r in nouns.items():
    if (r[0] <= 1000) and (len(word)>=2):
    #print('%8s:\t%.4f' % (word, r[0]))
        candi_words[word] = r[1]

tokenizer = LTokenizer(scores=candi_words)


a = data_tmp.clean_doc[0]

soytokens = tokenizer.tokenize(a)
tmp_words = list(set(soytokens))

doc_embedding = model.encode([a])
word_embedding = model.encode(tmp_words)

distances = cosine_similarity(doc_embedding, word_embedding)
mmr(doc_embedding, word_embedding, tmp_words, top_n=top_n, diversity=0.5)

tmp_keywords = mmr(doc_embedding, word_embedding, tmp_words, top_n=top_n, diversity=0.3)





#%%
word_list = []
for i in tqdm(range(len(data_tmp))):
    word_dict = {}
    
    word_dict['사고번호'] = str(data_tmp.사고번호[i])
    
    doc_ = data_tmp.clean_doc[i]
    tokens = tokenizer.tokenize(doc_)
    
    word_dict.update({'단어'+str(j): tokens[j] for j in range(len(tokens))})
    

