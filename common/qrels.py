import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any, Union

from ranking_utils.model.data import DataProcessor, ValTestDataset, ValTestInstance

def _qgroup_to_dict(group: pd.DataFrame, rel_type = int) -> dict[str, Any]:
	return {f"d{int(row['docid'])}": rel_type(row["relevancy"]) for _, row in group.iterrows()}

def load_qrels(datapath: Union[Path, str], rel_type = int) -> dict[str, dict[str, Any]]:
	if isinstance(datapath, str):
		datapath = Path(datapath)
	data = pd.read_csv(datapath, sep='\t', header=None)
	data.columns = ["queryid", "unused", "docid", "relevancy"]
	return {f"q{int(group)}": _qgroup_to_dict(gdata, rel_type) for group,gdata in tqdm(data.groupby(by="queryid"))}


class QrelDataset(ValTestDataset):
	def __init__(self, data_file: Union[Path, str], qrels_file: Union[Path, str], data_processor: DataProcessor) -> None:
		super().__init__(data_processor)
		self.data_file = Path(data_file)
		self.qrels_file = Path(qrels_file)
		self.qrels = pd.read_csv(self.qrels_file, sep='\t', header=None)
		self.qrels.columns = ["queryid", "unused", "docid", "relevancy"]
		self.fp = h5py.File(self.data_file, "r")
	
	def close(self) -> None:
		self.fp.close()

	def _num_instances(self) -> int:
		return len(self.qrels)

	def _get_instance(self, index: int) -> ValTestInstance:
		assert isinstance(index, int), "only a single instance can be retrieved at once"
		row = self.qrels.iloc[index]
		qid = row["queryid"].astype(int)
		docid = row["docid"].astype(int)
		label = row["relevancy"].astype(int)
		
		query = self.fp["queries"].asstr()[qid]
		doc = self.fp["docs"].asstr()[docid]
		return query, doc, qid, label
	
	def get_docid(self, index: int) -> int:
		row = self.qrels.iloc[[index]]
		return int(row["docid"])
