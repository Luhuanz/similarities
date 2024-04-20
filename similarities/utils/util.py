# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
copy from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
"""

import heapq
import queue
from typing import Union

import numpy as np
import torch
import torch.nn.functional

#计算两组向量之间的余弦相似度
def cos_sim(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    #它接受两个参数 a 和 b，这两个参数可以是 PyTorch 张量 (torch.Tensor) 或 NumPy 数组 (np.ndarray)。
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
#如果 a 是一维的（即形状为 (n,)），使用 unsqueeze(0)在第一个维度上增加一个维度，
 # 使其形状变为 (1, n)。这是为了确保即使输入是单个向量，也能以二维矩阵的形式进行处理
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)
#对 a 和 b 进行 L2 归一化，使每个向量的 L2 范数为 1。 p 的取值为 2 时，表示计算的是 L2 范数
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

#计算点积
def dot_score(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_dot_score(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i])
    :return: Vector with res[i] = dot_prod(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)


def pairwise_cos_sim(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
   Computes the pairwise cossim cos_sim(a[i], b[i])
   :return: Vector with res[i] = cos_sim(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def normalize_embeddings(embeddings: torch.Tensor):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


#使用余弦相似度来比较查询嵌入向量与一个语料库嵌入向量之间的相似性，通常用于信息检索或语义搜索。
# 它接受查询嵌入和语料库嵌入作为输入，并通过计算它们之间的余弦相似度来检索最相似的文档。
def semantic_search(
        query_embeddings: Union[torch.Tensor, np.ndarray],
        corpus_embeddings: Union[torch.Tensor, np.ndarray],
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000, #分别定义了查询和语料库的处理块大小
        top_k: int = 10,
        score_function=cos_sim
):
    """
    This function performs a cosine similarity search between a list of query embeddings and corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2-dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2-dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but
        requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed,
        but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys
        'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """
    #这段代码将输入的查询嵌入数据转换为 PyTorch 张量
    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    #确保查询嵌入和语料库嵌入在相同的设备上（CPU 或 GPU），如果不在，将查询嵌入移动到语料库嵌入所在的设备
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)
    #为每个查询嵌入初始化一个空列表，用于存储结果
    queries_result_list = [[] for _ in range(len(query_embeddings))]
    #计算相似度
    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarity
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # Get top-k scores 使用 torch.topk 为每个查询获取最高的 top_k 个相似度得分及其索引
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            #存储相似度得分遍历计算出的得分，将每个查询的最相似文档得分和对应的语料库索引存储在 queries_result_list 中。
            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(queries_result_list[query_id], (
                            score, corpus_id))  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {'corpus_id': corpus_id, 'score': score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x['score'],
                                               reverse=True)
#semantic_search 函数通过计算查询嵌入和语料库嵌入之间的余弦相似度，实现了语义搜索功能，可以用于信息检索或找到最相似的文档。
    return queries_result_list


#用于寻找嵌入空间中的相似句子对（即可能的释义对），主要基于余弦相似度计算。
def paraphrase_mining_embeddings(
        embeddings: torch.Tensor,
        query_chunk_size: int = 5000,
        corpus_chunk_size: int = 100000,
        max_pairs: int = 500000,
        top_k: int = 100,
        score_function=cos_sim
):
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param embeddings: A tensor with the embeddings
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """
#调整 top_k 因为每个句子与自身的相似度最高，初始化优先队列 pairs 用于存储最相似的句子对，min_score 用于跟踪队列中最低的分数。
    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0
#使用双重循环遍历嵌入张量，每次分别处理一个查询块和一个语料块。
    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            #对每个块对，计算块内所有句子之间的相似度分数
            scores = score_function(embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                    embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])
            #使用 torch.topk 获取每个查询句子相对于语料块中句子的最高 top_k 相似度分数及其索引
            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False)
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()
            #遍历分数矩阵，将每个句子对的索引计算出来。
            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr
            #如果句子对不是同一句子且分数高于 min_score，则将其添加到优先队列中。如果队列中的句子对数量达到 max_pairs，则移除最低分数的句子对
                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list  #对句子对列表按分数从高到低进行排序，并返回。

#找出相互接近（相似度高于给定阈值）的嵌入向量群。下面是对这个函数的逐行解释
def community_detection(embeddings, threshold=0.75, min_community_size=10, batch_size=10000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []

    # Maximum size for community 初始化存储提取社区的列表，调整最小社区大小，并设置排序时考虑的最大社区大小
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in range(0, len(embeddings), batch_size):
        # Compute cosine similarity scores 以批次方式处理嵌入向量，计算批次内嵌入向量与所有嵌入向量的余弦相似度
        cos_scores = cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                # Check if we need to increase sort_max_size
                while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                    sort_max_size = min(2 * sort_max_size, len(embeddings))
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                for idx, val in zip(top_idx_large.tolist(), top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        del cos_scores

    # Largest cluster first 按社区大小降序排序提取的社区
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities
