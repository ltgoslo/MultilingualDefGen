# coding=utf-8
# Copyright 2024 The GTE Team Authors and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import is_torch_npu_available


class GTEEmbeddidng(torch.nn.Module):
    def __init__(self,
                 model_name: str = None,
                 normalized: bool = True,
                 use_fp16: bool = True,
                 device: str = None
                ):
        super().__init__()
        self.normalized = normalized
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        self.use_fp16 = use_fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16 if self.use_fp16 else None
        )
        self.vocab_size = self.model.config.vocab_size
        self.model.to(self.device)

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: list):
        # conver to dict
        result = defaultdict(int)
        unused_tokens = set([self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                             self.tokenizer.unk_token_id])
        # token_weights = np.ceil(token_weights * 100)
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                token = self.tokenizer.decode([int(idx)])
                if w > result[token]:
                    result[token] = w
        return result

    @torch.no_grad()
    def encode(self,
               texts: None,
               dimension: int = None,
               max_length: int = 8192,
               batch_size: int = 16,
               return_dense: bool = True,
               return_sparse: bool = False):
        if dimension is None:
            dimension = self.model.config.hidden_size
        if isinstance(texts, str):
            texts = [texts]
        num_texts = len(texts)
        all_dense_vecs = []
        all_token_weights = []
        for n, i in enumerate(range(0, num_texts, batch_size)):
            batch = texts[i: i + batch_size]
            resulst = self._encode(batch, dimension, max_length, batch_size, return_dense, return_sparse)
            if return_dense:
                all_dense_vecs.append(resulst['dense_embeddings'])
            if return_sparse:
                all_token_weights.extend(resulst['token_weights'])
        all_dense_vecs = torch.cat(all_dense_vecs, dim=0)
        return {
            "dense_embeddings": all_dense_vecs,
            "token_weights": all_token_weights
        }

    @torch.no_grad()
    def _encode(self,
                texts: Dict[str, torch.Tensor] = None,
                dimension: int = None,
                max_length: int = 1024,
                batch_size: int = 16,
                return_dense: bool = True,
                return_sparse: bool = False):

        text_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        text_input = {k: v.to(self.model.device) for k,v in text_input.items()}
        model_out = self.model(**text_input, return_dict=True)

        output = {}
        if return_dense:
            dense_vecs = model_out.last_hidden_state[:, 0, :dimension]
            if self.normalized:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            output['dense_embeddings'] = dense_vecs
        if return_sparse:
            token_weights = torch.relu(model_out.logits).squeeze(-1)
            token_weights = list(map(self._process_token_weights, token_weights.detach().cpu().numpy().tolist(),
                                                    text_input['input_ids'].cpu().numpy().tolist()))
            output['token_weights'] = token_weights

        return output

    def _compute_sparse_scores(self, embs1, embs2):
        scores = 0
        for token, weight in embs1.items():
            if token in embs2:
                scores += weight * embs2[token]
        return scores

    def compute_sparse_scores(self, embs1, embs2):
        scores = [self._compute_sparse_scores(emb1, emb2) for emb1, emb2 in zip(embs1, embs2)]
        return np.array(scores)

    def compute_dense_scores(self, embs1, embs2):
        scores = torch.sum(embs1*embs2, dim=-1).cpu().detach().numpy()
        return scores

    @torch.no_grad()
    def compute_scores(self,
        text_pairs: List[Tuple[str, str]],
        dimension: int = None,
        max_length: int = 1024,
        batch_size: int = 16,
        dense_weight=1.0,
        sparse_weight=0.1):
        text1_list = [text_pair[0] for text_pair in text_pairs]
        text2_list = [text_pair[1] for text_pair in text_pairs]
        embs1 = self.encode(text1_list, dimension, max_length, batch_size, return_dense=True, return_sparse=True)
        embs2 = self.encode(text2_list, dimension, max_length, batch_size, return_dense=True, return_sparse=True)
        scores = self.compute_dense_scores(embs1['dense_embeddings'], embs2['dense_embeddings']) * dense_weight + \
            self.compute_sparse_scores(embs1['token_weights'], embs2['token_weights']) * sparse_weight
        scores = scores.tolist()
        return scores


if __name__ == '__main__':
    gte = GTEEmbeddidng('Alibaba-NLP/gte-multilingual-base')
    docs =  [
        "黑龙江离俄罗斯很近",
        "哈尔滨是中国黑龙江省的省会，位于中国东北",
        "you are the hero"
    ]
    print('docs', docs)
    embs = gte.encode(docs, return_dense=True,return_sparse=True)
    print('dense vecs', embs['dense_embeddings'])
    print('sparse vecs', embs['token_weights'])