#! /usr/bin/env python3

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig

import torch
from pytorch_lightning import seed_everything
from ranking_utils.model.data import TrainingMode
from ranking_utils.model import Ranker
import common
from common.qrels import QrelDataset, load_qrels

import h5py

import pandas as pd
from pathlib import Path
from typing import Union
from tqdm import tqdm

import pytrec_eval

def orig_qrel_to_qrel(data_file: Union[Path, str], qrels_file: Union[Path, str], out_file: str):
	qrels_file = Path(qrels_file)
	qrels = pd.read_csv(qrels_file, sep='\t', header=None)
	
	with h5py.File(data_file, "r") as fp:
		origqid_to_qid = {int(qid): idx for idx,qid in tqdm(enumerate(fp["orig_q_ids"]))}
		origdid_to_did = {int(did): idx for idx,did in tqdm(enumerate(fp["orig_doc_ids"]))}
	
	with open(out_file, "a") as out:
		for _, row in tqdm(qrels.iterrows()):
			qid = origqid_to_qid[int(row["queryid"])]
			docid = origdid_to_did[int(row["docid"])]
			label = int(row["relevancy"])
			out.write(f"{qid}\t0\t{docid}\t{label}\n")

def save_model_scores(model: Ranker, dataset: QrelDataset, out_path: Path, batch_size: int = 100):
	model.eval()
	dataset_size = len(dataset)
	with torch.inference_mode():
		with out_path.open('a') as out:
			for idx in tqdm(range(0, dataset_size, batch_size)):
				instances = [dataset[i] for i in range(idx, min(idx+batch_size, dataset_size))]

				batch, qids, labels = dataset.collate_fn(instances)

				batch = tuple([tensor.to(model.device) for tensor in batch])
				qids.to(model.device)
				labels.to(model.device)
				
				scores = model(batch).flatten().cpu().detach().numpy()

				for i, (_, qid, _) in enumerate(instances):
					docid = dataset.get_docid(idx+i)
					out.write(f"{qid}\t0\t{docid}\t{scores[i]}\n")

@hydra.main(config_path="hydra_conf", config_name="trec_eval", version_base=None)
def main(config: DictConfig):
	seed_everything(config.seed)
	common.set_cuda_devices_env(config.used_gpus)

	out_path = Path(config.out_path)
	data_dir = Path(config.data_path)#Path("/home/tim.hagen/datasets/TRECDL2019Passage/")
	data_file = data_dir / "data.h5"
	orig_qrels_file = data_dir / "qrels.tsv"
	qrels_file = data_dir / "qrels.mod.tsv"

	if not qrels_file.exists():
		print("Translating the qrels file to the doc- and query-ids used by the h5 converter")
		orig_qrel_to_qrel(data_file, orig_qrels_file, qrels_file)

	if not out_path.exists():
		print(f"Evaluating model and storing the results at {out_path}")
		model = common.load_model(config.model)
		assert isinstance(model, Ranker)
		model.training_mode = TrainingMode.POINTWISE
		model.to("cuda:0")
		dataprocessor = hydra_inst(config.model.dataprocessor)
		dataset = QrelDataset(data_file, qrels_file, dataprocessor)
		save_model_scores(model, dataset, out_path, batch_size=config.batch_size)
		dataset.close()
	else:
		print(f"Found results from a previous run at {out_path}")

	print("Loading the qrels...")
	qrels = load_qrels(qrels_file)
	print("Loading the model's scores...")
	run = load_qrels(out_path, float)
	print("Evaluating...")
	# relevance_level=2 since https://trec.nist.gov/data/deep2019.html
	evaluator: dict[str, dict[str, float]] = pytrec_eval.RelevanceEvaluator(qrels, {"map", "ndcg", "recip_rank"}, relevance_level=2)
	eval = evaluator.evaluate(run)
	df = pd.DataFrame([*eval.values()])
	print(df.mean())

if __name__ == '__main__':
	main()