import pandas as pd

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor

from sentence_transformers import SentenceTransformer

from utils import (
    load_data,
    cleaning,
    mmr,
    josa_delete
)


data_tmp = load_data()
data_tmp['clean_doc'] = data_tmp['doc'].map(lambda x: cleaning(x))


model = SentenceTransformer("distiluse-base-multilingual-cased-v1")


top_n = 3
n_gram_range = (1,2)

result_ = []

if __name__ == '__main__':
    for i in tqdm(range(len(data_tmp))):
        keywords_ = {}
        
        keywords_['사고번호'] = data_tmp.사고번호[i]
        doc_ = data_tmp.clean_doc[i]
        
        count = CountVectorizer(ngram_range = n_gram_range).fit([doc_])
        candidates = count.get_feature_names_out()
        
        doc_embedding = model.encode([doc_])
        candi_embedding = model.encode(candidates)
        
        # distances = cosine_similarity(doc_embedding, candi_embedding)
        tmp_keywords, keyword_sim = mmr(doc_embedding, candi_embedding, candidates, top_n=top_n, diversity=0.3)
        tmp_keywords = [josa_delete(x) for x in tmp_keywords]
        
        keywords_.update({'keyword' + str(j+1): tmp_keywords[j] for j in range(len(tmp_keywords))})
        keywords_.update({'keyword' + str(j+1) + 'score': keyword_sim[j] for j in range(len(keyword_sim))})
        
        result_.append(keywords_)

keywords = pd.DataFrame(keywords_)
keywords.to_excel('result/대표키워드.xlsx', header = True, index = False)

