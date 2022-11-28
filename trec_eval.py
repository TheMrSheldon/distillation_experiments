#! /usr/bin/env python3

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig

from ranking_utils.model.data.h5 import DataProcessor, H5PredictionDataset
from pytorch_lightning import Trainer, LightningModule, seed_everything

from torch.utils.data import DataLoader
from collections import defaultdict
import common
from common.trec_eval import trec_evaluation, load_qrels_from_file, load_run_from_file

import pandas as pd

from pathlib import Path
from tqdm import tqdm

from ranking_utils import write_trec_eval_file

@hydra.main(config_path="hydra_conf", config_name="trec_eval", version_base=None)
def main(config: DictConfig):
	config.model.model.hparams.bert_model = config.model.model.hparams.bert_model
	config.model.dataprocessor.bert_model = config.model.dataprocessor.bert_model

	seed_everything(config.seed)
	common.set_cuda_devices_env(config.used_gpus)
	out_path = Path(config.out_path)

	if not out_path.exists():
		print("Evaluating model")

		data_dir = Path(config.data_path)
		data_file = data_dir / "data.h5"
		test_file = data_dir / "fold_0" / "test.h5"

		dataprocessor = hydra_inst(config.model.dataprocessor)
		model = common.load_model(config.model)
		trainer = hydra_inst(config.trainer)

		assert isinstance(model, LightningModule)
		assert isinstance(dataprocessor, DataProcessor)
		assert isinstance(trainer, Trainer)
		assert trainer.devices == 1

		dataset = H5PredictionDataset(dataprocessor, data_file=data_file, pred_file_h5=test_file)
		dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, collate_fn=dataset.collate_fn)

		ids_iter = iter(dataset.ids())
		result = defaultdict(dict)
		
		predictions = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True)
		for item in tqdm(predictions):
			for score in item["scores"].detach().numpy():
				_, q_id, doc_id = next(ids_iter)
				result[q_id][doc_id] = score

		write_trec_eval_file(out_path, result, "test")
	else:
		print(f"Loading past evaluation run from file {out_path}")
		result = load_run_from_file(out_path)

	qrels = load_qrels_from_file(config.qrels_path)
	print(trec_evaluation(qrels, result, ["recip_rank", "map", "ndcg_cut.10"]))

if __name__ == '__main__':
	main()