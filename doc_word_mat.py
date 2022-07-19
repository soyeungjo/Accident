''' 
    단어문서행렬 생성
'''

import pandas as pd

from tqdm import tqdm

from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor

from sentence_transformers import SentenceTransformer

from utils import (
    load_data,
    cleaning,
)


data_tmp = load_data()
data_tmp['clean_doc'] = data_tmp['doc'].map(lambda x: cleaning(x))


noun_extractor = LRNounExtractor(verbose=False)
nouns = noun_extractor.train_extract(data_tmp.clean_doc)

candi_words = {} 
    
for word, r in nouns.items():
    if (r[0] <= 1000) and (len(word)>=2):
    #print('%8s:\t%.4f' % (word, r[0]))
        candi_words[word] = r[1]

tokenizer = LTokenizer(scores=candi_words)


word_list = []

for i in tqdm(range(len(data_tmp))):
    word_dict = {}
    
    word_dict['사고번호'] = str(data_tmp.사고번호[i])
    
    doc_ = data_tmp.clean_doc[i]
    tokens = tokenizer.tokenize(doc_)
    
    word_dict.update({'단어'+str(j): tokens[j] for j in range(len(tokens))})
    
    word_list.append(word_dict)

doc_word_mat = pd.DataFrame(word_list)
doc_word_mat.to_excel('result/단어문서행렬.xlsx', header = True, index = False)