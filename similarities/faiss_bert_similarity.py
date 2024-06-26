# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use faiss to build index and search
"""
import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import fire
import numpy as np
import pandas as pd
import requests
from loguru import logger
from requests.exceptions import RequestException
from text2vec import SentenceModel

from similarities.utils.util import cos_sim


def bert_embedding(
        input_dir: str,
        chunk_size: int = 10000,
        embeddings_dir: str = 'bert_engine/text_emb/',
        corpus_dir: str = 'bert_engine/corpus/',
        model_name: str = "shibing624/text2vec-base-chinese",
        batch_size: int = 32,
        target_devices: List[str] = None,
        normalize_embeddings: bool = False,
        text_column_name: str = "sentence",
        **kwargs,
):
    """
    Compute embeddings for a list of sentences
    :param input_dir: input dir, text files
    :param chunk_size: chunk size to save partial results
    :param embeddings_dir: save embeddings dir
    :param corpus_dir: save corpus file dir
    :param model_name: sentence emb model name
    :param batch_size: batch size
    :param target_devices: pytorch target devices, e.g. ['cuda:0', cuda:1]
        If None, all available CUDA devices will be used
    :param normalize_embeddings: whether to normalize embeddings before saving
    :param text_column_name: text column name, default sentence
    :param kwargs: read_csv kwargs, e.g. names=['sentence'], sep='\t'
    :return: None, save embeddings to embeddings_dir
    """
    #获取输入目录中的所有文本和 CSV 文件，确保至少存在一个文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith((".txt", ".csv"))]
    assert len(input_files) > 0, f"input_dir {input_dir} has no csv/txt files"
    logger.info(f"Start embedding, input files: {input_files}")
    #加载指定名称的句子嵌入模型。
    model = SentenceModel(model_name_or_path=model_name)
    logger.info(f'Load model success. model: {model_name}')
    #遍历输入文件，为每个文件创建一个路径，并初始化计数器用于输出文件名。
    for i, file in enumerate(input_files):
        logger.debug(f"Processing file {i + 1}/{len(input_files)}: {file}")
        output_file_counter = 0
        input_path = os.path.join(input_dir, file)
        # Read the input file in chunks 以块的形式读取文件，每块包含 chunk_size 行，从中提取出需要嵌入的文本。
        for chunk_df in pd.read_csv(input_path, chunksize=chunk_size, **kwargs):
            sentences = chunk_df[text_column_name].tolist()
            #在多个进程中计算嵌入，使用指定的设备和批处理大小，并根据需要归一化嵌入。
            pool = model.start_multi_process_pool(target_devices=target_devices)
            # Compute the embeddings using the multi processes pool
            emb = model.encode_multi_process(
                sentences,
                pool,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings
            )
            model.stop_multi_process_pool(pool)
            #确保嵌入目录存在，构建嵌入文件的路径，并将嵌入保存为 NumPy
            os.makedirs(embeddings_dir, exist_ok=True)
            text_embeddings_file = os.path.join(embeddings_dir, f"part-{output_file_counter:05d}.npy")
            np.save(text_embeddings_file, emb)
            logger.debug(f"Embeddings computed. Shape: {emb.shape}, saved to {text_embeddings_file}")

            # Save corpus to Parquet file 确保语料目录存在，构建语料文件的路径，并将当前块的数据保存为 Parquet 文件。
            os.makedirs(corpus_dir, exist_ok=True)
            corpus_file = os.path.join(corpus_dir, f"part-{output_file_counter:05d}.parquet")
            chunk_df.to_parquet(corpus_file, index=False)
            logger.debug(f"Corpus data saved to {corpus_file}")
            output_file_counter += 1
    logger.info(f"Embedding done, saved text emb to {embeddings_dir}")

#函数用于构建基于提供的文本嵌入的索引，通常是使用 FAISS 库通过 autofaiss 进行操作。 ：
def bert_index(
        embeddings_dir: str,
        index_dir: str = "bert_engine/text_index/",
        index_name: str = "faiss.index",
        max_index_memory_usage: str = "4G",
        current_memory_available: str = "8G",
        use_gpu: bool = False,
        nb_cores: Optional[int] = None,
):
    """
    Build indexes from text embeddings using autofaiss
    :param embeddings_dir: text embeddings dir, required
    :param index_dir: folder to save indexes, default `bert_engine/text_index/`
    :param index_name: indexes name to save, default `faiss.index`
    :param max_index_memory_usage: Maximum size allowed for the index, this bound is strict
    :param current_memory_available: Memory available on the machine creating the index, having more memory is a boost
        because it reduces the swipe between RAM and disk.
    :param use_gpu: whether to use gpu, default False
    :param nb_cores: Number of cores to use. Will try to guess the right number if not provided
    :return: None, save index to index_dir
    """
    from autofaiss import build_index  # pylint: disable=import-outside-toplevel

    if embeddings_dir and os.path.exists(embeddings_dir):
        logger.debug(f"Starting build index from {embeddings_dir}")
        logger.debug(
            f"Embedding path exist, building index "
            f"using embeddings {embeddings_dir} ; saving in {index_dir}"
        )
        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, index_name)
        index_infos_path = os.path.join(index_dir, index_name + ".json")
        try:
            build_index(
                embeddings=embeddings_dir,
                index_path=index_file,
                index_infos_path=index_infos_path,
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores,
                use_gpu=use_gpu,
            )
            logger.info(f"Index {embeddings_dir} done, saved in {index_file}, index infos in {index_infos_path}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Index {embeddings_dir} failed, {e}")
            raise e
    else:
        logger.warning(f"Embeddings dir {embeddings_dir} not exist")


def batch_search_index(
        queries,
        model,
        faiss_index,
        df,
        num_results,
        threshold,
        debug=False,
):
    """
    Search index with text inputs (batch search)
    :param queries: list of text
    :param model: sentence bert model
    :param faiss_index: faiss index
    :param df: corpus dataframe
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param debug: bool, whether to print debug info, default True
    :return: search results
    """
    result = []
    queries = [str(q).strip() for q in queries if str(q).strip()]
    #处理 queries，确保它们都是字符串，并移除两端的空白字符。
    if not queries:
        return result

    # Query embeddings need to be normalized for cosine similarity
    query_features = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    if query_features.shape[0] > 0:
        query_features = query_features.astype(np.float32)
        if threshold is not None:
            _, D, I = faiss_index.range_search(query_features, threshold)
        else:
            D, I = faiss_index.search(query_features, num_results)
            #遍历每个查询及其对应的距离 (d) 和索引 (i)，初始化一个用于存储每个查询结果的列表。
        for query, d, i in zip(queries, D, I):
            # Sorted faiss search result with distance
            text_scores = []
            #遍历每个查询的结果，从 df 中获取对应的文本，记录相似度得分和索引，并将其添加到结果列表中。
            for ed, ei in zip(d, i):
                sentence = df.iloc[ei].to_json(force_ascii=False)
                if debug:
                    logger.debug(f"query: {query}, Found: {sentence}, similarity: {ed}, id: {ei}")
                text_scores.append((sentence, float(ed), int(ei)))
            # Sort by score desc
            query_result = sorted(text_scores, key=lambda x: x[1], reverse=True)
            result.append(query_result)
    return result


def bert_filter(
        queries: List[str],
        output_file: str = "outputs/result.jsonl",
        model_name: str = "shibing624/text2vec-base-chinese",
        index_dir: str = 'bert_engine/text_index/',
        index_name: str = "faiss.index",
        corpus_dir: str = "bert_engine/corpus/",
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        debug: bool = False,
):
    """
    Entry point of bert filter, batch search index
    :param queries: list of texts, required
    :param output_file: save file path, default outputs/result.json
    :param model_name: clip model name
    :param index_dir: index dir, saved by bert_index, default bert_engine/image_index/
    :param index_name: index name, default `faiss.index`
    :param corpus_dir: corpus dir, saved by bert_embedding, default bert_engine/corpus/
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param device: pytorch device, e.g. 'cuda:0'
    :param debug: whether to print debug info, default False
    :return: batch search results
    """
    assert isinstance(queries, list), f"queries type error, queries: {queries}"

    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    faiss_index = faiss.read_index(index_file) #将语料库目录下的所有 Parquet 文件合并为一个 DataFrame
    model = SentenceModel(model_name_or_path=model_name, device=device)
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(Path(corpus_dir).glob("*.parquet")))
    logger.info(f'Load success. model: {model_name}, index: {faiss_index}, corpus size: {len(df)}')
    #使用 batch_search_index 函数进行批量搜索，传递上述加载的资源和参数
    result = batch_search_index(queries, model, faiss_index, df, num_results, threshold, debug=debug)
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for q, sorted_text_scores in zip(queries, result):
                json.dump(
                    {'query': q,
                     'results': [{'sentence': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
                    f,
                    ensure_ascii=False,
                )
                f.write('\n')
        logger.info(f"Query size: {len(queries)}, saved result to {output_file}")
    return result


def bert_server(
        model_name: str = "shibing624/text2vec-base-chinese",
        index_dir: str = 'bert_engine/text_index/',
        index_name: str = "faiss.index",
        corpus_dir: str = "bert_engine/corpus/",
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8001,
        debug: bool = False,
):
    '''
    model_name: 用于生成文本嵌入的 BERT 模型名称。
    index_dir: 存放 FAISS 索引的目录。
    index_name: FAISS 索引文件的名称。
    corpus_dir: 存放文本语料库的目录。
    num_results: 搜索时返回的结果数量。
    threshold: 搜索结果的相似度阈值。
    device: PyTorch 设备标识符。
    host: 服务的主机地址。
    port: 服务监听的端口。
    debug: 是否输出调试信息。
    '''
    """
    Main entry point of bert search backend, start the server
    :param model_name: sentence bert model name
    :param index_dir: index dir, saved by bert_index, default bert_engine/text_index/
    :param index_name: index name, default `faiss.index`
    :param corpus_dir: corpus dir, saved by bert_embedding, default bert_engine/corpus/
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param device: pytorch device, e.g. 'cuda:0'
    :param host: server host, default '0.0.0.0'
    :param port: server port, default 8001
    :param debug: whether to print debug info, default False
    :return: None, start the server
    """
    import uvicorn #与 Flask 框架不同，FastAPI 并不包含任何内置的开发服务器
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from starlette.middleware.cors import CORSMiddleware

    logger.info("starting boot of bert server")
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    faiss_index = faiss.read_index(index_file)
    model = SentenceModel(model_name_or_path=model_name, device=device)
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(Path(corpus_dir).glob("*.parquet")))
    logger.info(f'Load model success. model: {model_name}, index: {faiss_index}, corpus size: {len(df)}')

    # define the app 创建一个 FastAPI 应用，并添加 CORS 中间件以允许跨域请求。
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    class Item(BaseModel): #使用 Pydantic 定义请求数据模型，其中包含一个文本输入字段
        input: str = Field(..., max_length=512)

    @app.get('/')
    async def index():
        return {"message": "index, docs url: /docs"}

    @app.post('/emb') #定义/emb端点的POST请求，用于计算单个文本输入的嵌入
    async def emb(item: Item):
        try:
            q = item.input
            embeddings = model.encode(q)
            result_dict = {'emb': embeddings.tolist()}
            logger.debug(f"Successfully get sentence embeddings, q:{q}, res shape: {embeddings.shape}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/similarity') #定义/similarity端点的POST请求，计算两个文本输入的相似度。
    async def similarity(item1: Item, item2: Item):
        try:
            q1 = item1.input
            q2 = item2.input
            emb1 = model.encode(q1)
            emb2 = model.encode(q2)
            sim_score = cos_sim(emb1, emb2).tolist()[0][0]
            result_dict = {'similarity': sim_score}
            logger.debug(f"Successfully get similarity score, q1:{q1}, q2:{q2}, res: {sim_score}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/search') #定义/search端点的 POST 请求，根据单个文本查询执行搜索，并返回最相关的结果
    async def search(item: Item):
        try:
            q = item.input
            results = batch_search_index([q], model, faiss_index, df, num_results, threshold, debug=debug)
            # input is one query, so get first result
            sorted_text_scores = results[0]
            result_dict = {'result': sorted_text_scores}
            logger.debug(f"Successfully search done, q:{q}, res size: {len(sorted_text_scores)}")
            return result_dict
        except Exception as e:
            logger.error(f"search error: {e}")
            return {'status': False, 'msg': e}, 400

    logger.info("Server starting!")
    uvicorn.run(app, host=host, port=port)


class BertClient:
    def __init__(self, base_url: str = "http://localhost:6007", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout

    def _post(self, endpoint: str, data: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}/{endpoint}", json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Request failed: {e}")
            return {}

    def get_emb(self, input_text: str) -> List[float]:
        try:
            data = {"input": input_text}
            response = self._post("emb", data)
            return response.get("emb", [])
        except Exception as e:
            logger.error(f"get_emb error: {e}")
            return []

    def get_similarity(self, input_text1: str, input_text2: str) -> float:
        try:
            data1 = {"input": input_text1}
            data2 = {"input": input_text2}
            response = self._post("similarity", {"item1": data1, "item2": data2})
            return response.get("similarity", 0.0)
        except Exception as e:
            logger.error(f"get_similarity error: {e}")
            return 0.0

    def search(self, input_text: str):
        try:
            data = {"input": input_text}
            response = self._post("search", data)
            return response.get("result", [])
        except Exception as e:
            logger.error(f"search error: {e}")
            return []


def main():
    """Main entry point"""
    fire.Fire(
        {
            "bert_embedding": bert_embedding,
            "bert_index": bert_index,
            "bert_filter": bert_filter,
            "bert_server": bert_server,
        }
    )


if __name__ == "__main__":
    main()
