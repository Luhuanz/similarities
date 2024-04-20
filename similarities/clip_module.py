# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLIP model for text and image embeddings
"""
import math
import queue
from typing import List, Union, Dict

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional
from PIL import Image
from loguru import logger #Loguru 是一个功能丰富且易于使用的日志记录
from torch import nn
from tqdm import trange
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, CLIPProcessor, CLIPModel


class ClipModule(nn.Module):
    """
    CLIP model for text and image embeddings

    Args:
        model_name_or_path: str, default "OFA-Sys/chinese-clip-vit-base-patch16"
            chinese model url: https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16
            english model url: https://huggingface.co/openai/clip-vit-base-patch32
        processor_name: str, default None
        device: str, default None
        is_chinese_model: bool, default None, if None, auto detect by model_name_or_path
    """

    def __init__(
            self,
            model_name_or_path: str = "OFA-Sys/chinese-clip-vit-base-patch16",
            processor_name: str = None,
            device: str = None,
            is_chinese_model: bool = None,
    ):
        super(ClipModule, self).__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_name_or_path = model_name_or_path
        if processor_name is None:
            processor_name = model_name_or_path
        if is_chinese_model is None:
            is_chinese_model = 'chinese' in model_name_or_path
        self.is_chinese_model = is_chinese_model
        if is_chinese_model:
            self.model = ChineseCLIPModel.from_pretrained(model_name_or_path)
            self.processor = ChineseCLIPProcessor.from_pretrained(processor_name)
        else:
            self.model = CLIPModel.from_pretrained(model_name_or_path)
            self.processor = CLIPProcessor.from_pretrained(processor_name) # 对数据进行预处理
        logger.debug(f"Device: {self.device}")
        self.model.to(self.device)

    def __str__(self):
        return f"model_name_or_path: {self.model_name_or_path} ClipModule({self.model})"

    def forward(self, features):

        image_embeds = []
        text_embeds = []

        if 'pixel_values' in features:  #这pixel_values种预处理可能包括归一化、调整大小等，使得图像数据符合模型的输入要求。
            vision_outputs = self.model.vision_model(pixel_values=features['pixel_values'])#[3,768]
            image_embeds = self.model.visual_projection(vision_outputs[1]) #[3,512]
            #通常是最后一层的输出或特定的中间层输出）传递给 self.model.visual_projection。visual_projection 这部分通常是一个线性层或其他形式的投影层，用于将模型的视觉输出转换为更紧凑或特定维度的嵌入表示（image_embeds）

        if 'input_ids' in features: #处理文本输入并提取特征,说明有文本数据需要处理。'input_ids' 是经过编码的文本数据，通常是一个整数数组，每个整数代表词汇表中的一个单词或标记。
            text_outputs = self.model.text_model(
                input_ids=features.get('input_ids'),
                attention_mask=features.get('attention_mask', None), #指定哪些部分是真实文本，哪些部分是填充的。这可以帮助模型区分真实数据和填充数据。
                position_ids=features.get('position_ids', None),
                output_attentions=features.get('output_attentions', None),
                output_hidden_states=features.get('output_hidden_states', None),#根据需要返回模型的注意力和隐藏状态。这个调用返回 text_outputs，它包含了文本处理的结果。
            )
            if self.is_chinese_model:
                # refer chinese clip: https://github.com/huggingface/transformers/blob/main/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1431
                pooled_output = text_outputs[0][:, 0, :]
            else:
                pooled_output = text_outputs[1]
            text_embeds = self.model.text_projection(pooled_output)

        sentence_embedding = []
        image_features = iter(image_embeds)
        text_features = iter(text_embeds)

        for idx, input_type in enumerate(features['image_text_info']):
            if input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features['embedding'] = torch.stack(sentence_embedding).float()

        return features

    def tokenize(self, texts):
        images = []
        texts_values = []
        image_text_info = []

        for idx, data in enumerate(texts):
            if isinstance(data, (Image.Image, np.ndarray)):  # An Image
                images.append(data)
                image_text_info.append(0)
            else:  # A text
                texts_values.append(data)
                image_text_info.append(1)

        if len(texts_values) == 0:
            texts_values = None
        if len(images) == 0:
            images = None
#images=[<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=282x288 at 0x7FE91D2BA430>, <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=333x249 at 0x7FE91D2BA490>, <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=746x520 at 0x7FE91CB342E0>]
        inputs = self.processor(text=texts_values, images=images, return_tensors="pt", padding=True)
        inputs['image_text_info'] = image_text_info
        return inputs

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str):
        return ClipModule(model_name_or_path=input_path)

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    @staticmethod
    def batch_to_device(batch, device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch

    def encode(
            self,
            sentences: Union[str, List[str], Image.Image, List[Image.Image]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            device: str = None,
    ):
        """
        Computes sentence and images embeddings

        :param sentences: list of sentences or list of Image.Image
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, the output is a stacked tensor. Else, it is a list of pytorch tensors.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
            In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :param device: Which device to use for the computation

        :return:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        if device is None:
            device = self.device
        self.model.eval()
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #这行代码检查 sentences 是否为字符串或者是否没有 __len__ 属性（
            sentences = [sentences]
            input_was_string = True
        self.model.to(device)
        if convert_to_tensor:
            convert_to_numpy = False
        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sent) for sent in sentences])#计算了 sentences 中每个句子的长度，并根据长度对句子进行排序
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = self.batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features['embedding']
                embeddings = embeddings.detach() #用来从当前计算图中分离出 embeddings 张量 detach() 方法会创建一个新的张量，它与原始张量共享数据但不会记录梯度
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        else:
            all_embeddings = torch.stack(all_embeddings)
        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi processes to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices,
            list, e.g. ['cuda:0', 'cuda:1'] If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu'] * 4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))
#模拟多台设备这里使用 'spawn' 方式来启动进程。不同的操作系统支持不同的进程启动方式，'spawn' 是跨平台的，意味着它在 Windows、macOS 和 Unix 上都可用
        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=ClipModule._encode_multi_process_worker, # encodeing
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True
            )
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()

    def encode_multi_process(
            self,
            sentences: Union[List[str], List[Image.Image]],
            pool: Dict[str, object],
            batch_size: int = 32,
            normalize_embeddings: bool = False,
            chunk_size: int = None
    ):
        # 在多个 GPU 上并行地编码大量句子或图像。
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences, or list of images
        :param pool: A pool of workers started with start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param normalize_embeddings: bool, Whether to normalize embeddings before returning them
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it is a sensible size.
        :return: Numpy matrix with all embeddings
        """
#没有指定 chunk_size（每个数据块的大小），则自动计算一个合理的值。它试图将句子均匀分配到各个进程，但每个块的大小不超过 5000
        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")
      #  获取进程池中用于输入的队列
        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []
#初始化 last_chunk_id 来跟踪当前的块编号，初始化 chunk 列表来收集每个数据块的
        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, normalize_embeddings])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, normalize_embeddings])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi processes setup
        """
        while True:
            try:
                id, batch_size, sentences, normalize_embeddings = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    device=target_device,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                )
                results_queue.put([id, embeddings])
            except queue.Empty:
                break
