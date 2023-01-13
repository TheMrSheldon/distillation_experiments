
from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchtext.vocab import GloVe
from transformers import get_constant_schedule

from typing import Any, Iterable, Optional, Tuple

Input = Tuple[Tensor, Tensor]
Batch = Tuple[Tensor, Tensor, Optional[Tensor]]

_stopwords = ['', ',', '.', '\'s', '"', '-', '?', '!', '/', '(', ')', '_', 'the', 'be', 'to', 'of', 'and', 'a',
					   'an', 'in', 'that', 'it', 'you', 'me', 'i', 'is', 'at']

def split_and_pad_words(text: str, result_length: int, padding: str="") -> list[str]:
	toks = list(set([tok for tok in text.split(' ')]))
	if len(toks) > result_length:
		return toks[:result_length]
	return toks + [padding]*(result_length - len(toks))

class GloVeEmbedding:
	def __init__(self, embedding_dim: int=300) -> None:
		self.embedding_dim = embedding_dim
		self.glove = GloVe()

	def word_embedding(self, words: list[str]) -> torch.FloatTensor:
		embedding = torch.zeros(len(words), self.embedding_dim)
		for i, txt in enumerate(words):
			embedding[i] = self.glove[txt]
		return embedding

class GraphOfWordAdj:
	def __init__(self, doc_len: int, stopwords=_stopwords, window_size: int = 5) -> None:
		self.doc_len = doc_len
		self.stopwords = stopwords
		self.embedding = GloVeEmbedding()
		self.window_size = window_size

	def __call__(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
		text_split = [token for token in text.split(' ') if token not in self.stopwords]
		#tokens = list(set(text_split))
		tokens = split_and_pad_words(text, self.doc_len)

		x = self.embedding.word_embedding(tokens)
		
		adj = torch.zeros(len(tokens), len(tokens))
		for d in range(1, self.window_size+1):
			for txt_1, txt_2 in zip(text_split[:-d], text_split[d:]):
				idx_1 = tokens.index(txt_1)
				idx_2 = tokens.index(txt_2)
				adj[idx_1, idx_2] = 1

		return (x, adj)

class GRMMDataProcessor(DataProcessor):
	def __init__(self, qrl_len: int = 20, doc_len: int = 300) -> None:
		super().__init__()
		self.qrl_len = qrl_len
		self.doc_len = doc_len
		self.gow = GraphOfWordAdj(doc_len, stopwords=[])
	
	def get_model_input(self, query: str, doc: str) -> Input:
		q_tok = split_and_pad_words(query, self.qrl_len)
		d_tok = split_and_pad_words(doc, self.doc_len)

		_, adj = self.gow(doc)
		q_emb = self.gow.embedding.word_embedding(q_tok)
		d_emb = self.gow.embedding.word_embedding(d_tok)
		x = self.create_simmat(q_emb, d_emb).permute(1,0)
		return x, adj

	def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
		inputs = list(inputs)
		return torch.stack([x for x,_ in inputs]), torch.stack([adj for _,adj in inputs]), None

	def create_simmat(self, a_emb: torch.Tensor, b_emb: torch.Tensor) -> torch.Tensor:
		A, B = a_emb.shape[0], b_emb.shape[0]
		a_denom = a_emb.norm(p=2, dim=1).reshape(A, 1).expand(A, B) + 1e-9 # avoid 0div
		b_denom = b_emb.norm(p=2, dim=1).reshape(1, B).expand(A, B) + 1e-9 # avoid 0div
		perm = b_emb.permute(1, 0)
		sim = torch.mm(a_emb, perm)
		sim = sim / (a_denom * b_denom)
		return sim


class GRMMRanker(Ranker):
	def __init__(self, lr: float, layers: int = 2, topk: int = 40, qrl_len: int = 20, dropout: float = 0, idf: bool = False) -> None:
		super().__init__()
		self.lr = lr
		self.layers = layers
		self.topk = topk
		self.idf_flag = idf

		self.linear1 = nn.Linear(topk, 64)
		self.linear2 = nn.Linear(64, 32) 
		self.linear3 = nn.Linear(32, 1) 
		self.linearz0 = nn.Linear(qrl_len, qrl_len) 
		self.linearz1 = nn.Linear(qrl_len, qrl_len) 
		self.linearr0 = nn.Linear(qrl_len, qrl_len) 
		self.linearr1 = nn.Linear(qrl_len, qrl_len) 
		self.linearh0 = nn.Linear(qrl_len, qrl_len) 
		self.linearh1 = nn.Linear(qrl_len, qrl_len) 
		#self.gated = nn.Linear(1, 1) 
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, batch: Batch) -> torch.Tensor:
		x, adj, idf = batch
		feat = x
		for _ in range(self.layers):
			x = feat
			a = adj.matmul(x)

			z0 = self.linearz0(a)
			z1 = self.linearz1(x)
			z = F.sigmoid(z0 + z1)

			r0 = self.linearr0(a)
			r1 = self.linearr1(x)
			r = F.sigmoid(r0 + r1)

			h0 = self.linearh0(a)
			h1 = self.linearh1(r*x)
			h = F.relu(h0 + h1)

			feat = h*z + x*(1-z)
			x = self.dropout(feat)


		feat = feat.permute(0, 2, 1)
		topk, _ = feat.topk(self.topk, -1) # batch, qrl, doc
		rel = F.relu(self.linear1(topk))
		rel = F.relu(self.linear2(rel))
		rel = self.linear3(rel)
		#if self.idf_flag:
		#	assert idf is not None
		#	gated_weight = F.softmax(self.gated(idf), dim=1)
		#	rel = rel * gated_weight
		scores = rel.squeeze(-1).sum(-1, keepdim=True)
		return scores
	
	def configure_optimizers(self) -> Tuple[list[Any], list[Any]]:
		opt = torch.optim.AdamW(
			filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
		)
		sched = get_constant_schedule(opt)
		return [opt], [{"scheduler": sched, "interval": "step"}]