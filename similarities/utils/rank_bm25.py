# -*- coding: utf-8 -*-
# Author: dorianbrown
# Brief: https://github.com/dorianbrown/rank_bm25

import math
from multiprocessing import Pool, cpu_count
from collections import Counter

import numpy as np

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined
Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0 #avgdl 为文档的平均长度
        self.doc_freqs = [] #个文档的词频统计
        self.idf = {} #词项的逆文档频率
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer: #对语料库中的每个文档进行分词。
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)#_initialize 方法初始化文档频率和文档长度等数据，
        self._calc_idf(nd) #调用 _calc_idf 方法计算逆文档频率（IDF）

    def _initialize(self, corpus):
        #遍历语料库中的每个文档，计算并更新文档长度、词频和词的文档频率。
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = dict(Counter(document))
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):#如果提供了 tokenizer，则对 corpus 中的每个文档进行分词。
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):#用于计算查询与语料库文档的相似度分数
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5): #根据计算得到的分数，返回与查询最相关的前 n 个文档。

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

#https://www.cnblogs.com/novwind/p/15177871.html#23-%E5%AD%97%E6%AE%B5%E9%95%BF%E5%BA%A6%E4%B8%8E%E5%B9%B3%E5%9D%87%E9%95%BF%E5%BA%A6%E9%83%A8%E5%88%86
class BM25Okapi(BM25):
    # 将 k1、b 和 epsilon 存储为实例变量，并调用基类 BM25 的构造函数。
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    #覆盖基类的 _calc_idf 方法来计算逆文档频率（IDF）
    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        # 初始化 IDF 总和变量 idf_sum 和负 IDF 值列表 negative_idfs
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
       #遍历每个词及其在文档中出现的频率，计算每个词的IDF值，并累加到 idf_sum。如果 IDF 值为负，则将词添加到 negative_idfs。
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        #计算平均 IDF 值。
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query: str
        :return: array
        """
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            scores += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))) #计算公式
        return scores


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            scores += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return scores


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            scores += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return scores
