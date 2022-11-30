#! /usr/bin/env python3

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from ranking_utils.model.data.h5 import H5DataModule
import common

@hydra.main(config_path="hydra_conf", config_name="train", version_base=None)
def main(config: DictConfig):
	distil = config.get("teacher") is not None
	#if distil:
	#	config.teacher.model.hparams.bert_model = config.teacher.model.hparams.bert_model
	#	config.teacher.dataprocessor.bert_model = config.teacher.dataprocessor.bert_model
	#config.model.model.hparams.bert_model = config.model.model.hparams.bert_model
	#config.model.dataprocessor.bert_model = config.model.dataprocessor.bert_model

	seed_everything(config.seed)
	common.set_cuda_devices_env(config.used_gpus)
	checkpoint_path = config.checkpoint_path
	
	model = student = common.load_model(config.model)
	teacher = None if not distil else common.load_model(config.teacher)
	datamodule = hydra_inst(config.datamodule, data_processor=config.model.dataprocessor)

	assert isinstance(model, LightningModule)
	assert isinstance(student, LightningModule)
	assert isinstance(teacher, LightningModule) or teacher is None
	assert isinstance(datamodule, H5DataModule)

	checkpointcb = ModelCheckpoint(
		dirpath=checkpoint_path,
		save_top_k=-1,
		filename="epoch{epoch:02d}"
	)
	trainer = hydra_inst(config.trainer, callbacks=[checkpointcb])

	assert isinstance(trainer, Trainer)

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