
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os
content = ['How to format mu hard disk','Hard disk format problems']
'''
切分文本
停词处理，
词频统计,向量化
词频归一化，
词干处理
词频统计中计算TF-IDF

'''
import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):#词频统计向量化
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))



class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def countVectorizerDataSet(path,content):
    '''

    :param path:
    :param content:
    :return: 向量化的词频统计
    '''
    vectorizer = StemmedCountVectorizer(min_df=1,stop_words='english',decode_error='ignore')#此处可以设置stop_words='English',具体单词可以get_stop_words()
    X = vectorizer.fit_transform(content)
    feature_names = vectorizer.get_feature_names()

    return feature_names,X.toarray().transpose()

path = '/Users/huihui/PycharmProjects/ml_datamining/Ch04/email/ham'
posts = [open(os.path.join(path,file),errors='ignore').read() for file in os.listdir(path)]

res1,res2 = countVectorizerDataSet(path,posts)
print(res1,res2)




import scipy as sp
def dist_norm(v1,v2):
    '''

    :param v1: vect1
    :param v2: vect2
    :return: 归一化之后的距离
    '''
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())#,scipy的linalg模块用于线性代数，norm计算欧几里得范式
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())





'''
边角料
'''
import nltk
def wordProcessingWithNltk(words):
    '''

    :param words:
    :return: 词干处理，包括前后缀过去式等等
    '''
    s = nltk.stem.SnowballStemmer('english')
    return s.stem(words)

import scipy as sp
import math
def tfidf(term,doc,docset):
    tf = float(doc.count(term)) / sum(doc.count(w) for w in docset)
    idf = math.log(float(len(docset))/len([doc for doc in docset if term in doc]))
    return tf * idf

