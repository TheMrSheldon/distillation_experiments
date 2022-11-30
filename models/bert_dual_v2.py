from typing import Any, Iterable, Tuple

from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor

import torch
from transformers import DistilBertModel, DistilBertTokenizer, get_constant_schedule_with_warmup

Input = Tuple[str, str]

BERT_Batch = Tuple[torch.LongTensor, torch.LongTensor]
Batch = Tuple[BERT_Batch, BERT_Batch]

class BERTDualv2DataProcessor(DataProcessor):
	def __init__(self, bert_doc_model: str, bert_q_model: str, char_limit: int) -> None:
		super().__init__()
		self.doc_tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(bert_doc_model)
		self.query_tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(bert_q_model)
		self.char_limit = char_limit
	
	def get_model_input(self, query: str, doc: str) -> Input:
		# empty queries or documents might cause problems later on
		if len(query.strip()) == 0:
			query = "(empty)"
		if len(doc.strip()) == 0:
			doc = "(empty)"

		# limit characters to avoid tokenization bottlenecks
		return query[: self.char_limit], doc[: self.char_limit]

	def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
		queries, docs = zip(*inputs)
		query_in = self.query_tokenizer(queries, padding=True, truncation=True)
		doc_in = self.doc_tokenizer(docs, padding=True, truncation=True)
		return (
			(torch.LongTensor(query_in["input_ids"]), torch.LongTensor(query_in["attention_mask"])),
			(torch.LongTensor(doc_in["input_ids"]), torch.LongTensor(doc_in["attention_mask"])),
		)

class BERTDualv2Ranker(Ranker):
	def __init__(self, lr: float, warmup_steps: int, hparams: dict[str, Any]) -> None:
		super().__init__()
		self.lr = lr
		self.warmup_steps = warmup_steps
		self.save_hyperparameters(hparams)

		self.bert_doc: DistilBertModel = DistilBertModel.from_pretrained(hparams["bert_doc_model"], return_dict=True)
		for p in self.bert_doc.parameters():
			p.requires_grad = not hparams["freeze_bert_doc"]
			
		self.bert_q: DistilBertModel = DistilBertModel.from_pretrained(hparams["bert_q_model"], return_dict=True)
		for p in self.bert_doc.parameters():
			p.requires_grad = not hparams["freeze_bert_q"]
		
		self.dropout = torch.nn.Dropout(hparams["dropout"])
		#self.classification = torch.nn.Linear(
		#	self.bert_doc.config.hidden_size + self.bert_q.config.hidden_size, 1
		#)
		self.classification = torch.nn.Sequential(
			torch.nn.Linear(self.bert_doc.config.hidden_size + self.bert_q.config.hidden_size, 500),
			torch.nn.Sigmoid(),
			torch.nn.Linear(500, 1)
		)

	def forward(self, batch: Batch) -> torch.Tensor:
		doc_in, query_in = batch
		doc_cls_out = self.bert_doc(*doc_in)["last_hidden_state"][:, 0]
		query_cls_out = self.bert_q(*query_in)["last_hidden_state"][:, 0]
		cls_out = torch.cat((doc_cls_out, query_cls_out), dim=1)
		return self.classification(self.dropout(cls_out))
	
	def configure_optimizers(self) -> Tuple[list[Any], list[Any]]:
		opt = torch.optim.AdamW(
			filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
		)
		sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
		return [opt], [{"scheduler": sched, "interval": "step"}]
