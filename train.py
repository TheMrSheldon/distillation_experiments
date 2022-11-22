#! /usr/bin/env python3

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig
import os
from pytorch_lightning import LightningModule, Trainer, seed_everything
from ranking_utils.model.data.h5 import H5DataModule
import common

@hydra.main(config_path="hydra_conf", config_name="train", version_base=None)
def main(config: DictConfig):
	distil = config.get("teacher") is not None
	if distil:
		config.teacher.model.hparams.bert_model = config.teacher.model.hparams.bert_model
		config.teacher.dataprocessor.bert_model = config.teacher.dataprocessor.bert_model
	config.model.model.hparams.bert_model = config.model.model.hparams.bert_model
	config.model.dataprocessor.bert_model = config.model.dataprocessor.bert_model

	seed_everything(config.seed)
	common.set_cuda_devices_env(config.used_gpus)
	
	model = student = common.load_model(config.model)
	teacher = None if not distil else common.load_model(config.teacher)
	datamodule = hydra_inst(config.datamodule, data_processor=config.model.dataprocessor)
	trainer = hydra_inst(config.trainer)

	assert isinstance(model, LightningModule)
	assert isinstance(student, LightningModule)
	assert isinstance(datamodule, H5DataModule)
	assert isinstance(trainer, Trainer)
	assert isinstance(teacher, LightningModule) or teacher is None

	if teacher is not None:
		from distillation import DistillationRanker
		model = DistillationRanker(student=student, teacher=teacher, datamodule=datamodule)
		print("Switching to distillation mode")
		print(f"\tTeacher Model: {teacher._get_name()}")
		print(f"\tStudent Model: {student._get_name()}")

	checkpoint = config.checkpoint
	if checkpoint is not None:
		print(f"Resuming from checkpoint at {checkpoint}")
	trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)

if __name__ == '__main__':
	main()