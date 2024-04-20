# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import jieba
import jieba.posseg #jieba.posseg 用于带词性标注的分词。

from jieba.analyse.tfidf import DEFAULT_IDF, _get_abs_path #导入 TF-IDF 相关的默认 IDF 值和获取绝对路径的函数。
## 获取当前脚本所在目录的绝对路径
pwd_path = os.path.abspath(os.path.dirname(__file__))
#设置默认停用词文件的路径。假定停用词文件位于代码文件同级的 data 目录下
default_stopwords_file = os.path.join(pwd_path, '../data/stopwords.txt')

#用于从给定路径加载停用词
def load_stopwords(file_path):
    stopwords = set()
    if file_path and os.path.exists(file_path):  #检查提供的 file_path 是否有效并且文件存在
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  #去除每行的前后空白字符，并将其添加到 stopwords 集合中
                stopwords.add(line)
    return stopwords


class IDFLoader: #用于加载和处理 IDF（逆文档频率）值
    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {} #idf_freq为字典，存储每个词的IDF值
        self.median_idf = 0.0 #median_idf存储IDF值的中位数。
        if idf_path:
            self.set_new_path(idf_path)#加载 IDF 文件

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {} #清空 idf_freq 字典，然后逐行解析IDF文件的内容，将每个词及其IDF值存储到idf_freq字典中。
            for line in content.splitlines(): #函数将文本内容分割成行，并通过迭代来处理每一行
                word, freq = line.strip().split(' ')#对于每一行，使用 strip() 函数去除行首行尾的空白字符，然后使用 split(' ') 函数以空格为分隔符将行拆分成单词和频率。拆分后的结果赋值给 word 和 freq。
                self.idf_freq[word] = float(freq) #将单词和其频率存储到 self.idf_freq 字典中
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2] #这一行的目的是计算频率中位数。它首先使用 values() 方法获取 self.idf_freq 字典中的频率值，并通过 sorted() 函数将这些频率值进行排序。

    def get_idf(self):
        return self.idf_freq, self.median_idf #返回 IDF 频率字典和中位数 IDF 值


class TFIDF:
    def __init__(self, idf_path=None, stopwords=None):
        self.stopwords = stopwords if stopwords is not None else load_stopwords(default_stopwords_file) #如果提供了 stopwords 参数就使用它，否则从默认的停用词文件中加载。
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF) #创建 IDFLoader 的实例来加载 IDF 值，如果没有提供 idf_path，则使用默认的 IDF 值
        self.idf_freq, self.median_idf = self.idf_loader.get_idf() #从 idf_loader 中获取 IDF 频率字典和中位数 IDF 值。

    def set_idf_path(self, idf_path): #用于更新 IDF 文件路径
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("IDF file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def get_tfidf(self, sentence):
    #使用jieba.posseg.cut 对句子进行分词和词性标注，并筛选出除去一些特定词性（如助词、标点符号等）的词。
        words = [word.word for word in jieba.posseg.cut(sentence) if word.flag[0] not in ['u', 'x', 'w']]
     #进一步过滤掉停用词和长度小于 2 的词。
        words = [word for word in words if word.lower() not in self.stopwords or len(word.strip()) < 2]
     #为句子中的每个词计算IDF值，如果词不在IDF字典中，则使用中位数IDF值
        word_idf = {word: self.idf_freq.get(word, self.median_idf) for word in words}

        res = []
    #创建一个结果列表res，并为IDF字典中的每个词填充其TF-IDF值，如果词不在word_idf 中，则其值为0。
        for w in list(self.idf_freq.keys()):
            res.append(word_idf.get(w, 0))
        return res
