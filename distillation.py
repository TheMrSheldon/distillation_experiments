from typing import Any
from pytorch_lightning import LightningModule
from ranking_utils.model import Ranker, TrainingBatch, TrainingMode
from ranking_utils.model.data.h5 import H5DataModule
import torch

class DistillationRanker(LightningModule):

	def __init__(self, student: Ranker, teacher: Ranker, datamodule: H5DataModule, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.mse = torch.nn.MSELoss()
		self.student = student
		self.teacher = teacher
		for p in self.teacher.parameters():
			p.requires_grad = False
		
		self.student.training_mode = TrainingMode.PAIRWISE
		self.teacher.training_mode = TrainingMode.PAIRWISE
		self.teacher.eval()
		datamodule.training_mode = TrainingMode.PAIRWISE

	def training_step(
		self,
		batch: TrainingBatch,
		batch_idx: int,
	) -> torch.Tensor:
		"""Train a single batch.

		Args:
			batch (TrainingBatch): A training batch.
			batch_idx (int): Batch index.

		Returns:
			torch.Tensor: Training loss.
		"""
		pos_model_batch, neg_model_batch, _ = batch

		# Student outputs:
		student_pos = self.student(pos_model_batch)
		student_neg = self.student(neg_model_batch)

		# Teacher outputs:
		teacher_pos = self.teacher(pos_model_batch)
		teacher_neg = self.teacher(neg_model_batch)

		# Compute Margin-MSE (https://arxiv.org/pdf/2010.02666.pdf)
		student_diff = student_pos - student_neg
		teacher_diff = teacher_pos - teacher_neg
		loss = self.mse(student_diff, teacher_diff)

		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		return self.student.configure_optimizers()