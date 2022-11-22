
<!--
Check what GPUs are currently unused via `nvidia-smi`.
Let X be an available GPU.

# Training
Example for training BERT on TRECDL2019Passage folder 0 with a batch size of 20. Note that `X` should be replaced with the desired cuda device.
```
./train.py \
	model=bert_dot \
	used_gpus=[2,3] \
	trainer.devices=2 \
	datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=20 \
	model.hparams.freeze_bert=True
```

Distillation:
```
./train.py \
	model=distilbert_dot \
	model@teacher=bert_dot \
	used_gpus=[3] \
	datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=20
```

Distill from Checkpoint of teacher:
```
./train.py \
	model=distilbert_dot \
	model@teacher=bert_dot \
	teacher.checkpoint="./checkpoints/bert_refine/epoch\=0-step\=44247.ckpt" \
	used_gpus=[1,2,3] \
	trainer.devices=3 \
	datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=20 \
	trainer.limit_val_batches=0.1 \
	trainer.max_epochs=2
```

# Evaluating
```
./validation.py \
	model=distilbert_dot \
	model@teacher=bert_dot \
	used_gpus=[2] \
	trainer.devices=1 \
	trainer.default_root_dir="./checkpoints/" \
	datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=200 \
	checkpoint="./checkpoints/lightning_logs/version_3/checkpoints/epoch\=0-step\=13273.ckpt" \
	trainer.limit_val_batches=0.1
```





# Results (Dataset: TRECDL2019 Passage)
### BERT (untrained):
 Metric                    | Score
---------------------------|----------------------
val_RetrievalMAP           | 0.007322873920202255
val_RetrievalMRR           | 0.007324859965592623
val_RetrievalNormalizedDCG | 0.0991392657160759

### BERT (1 epoch):
 Metric                    | Score
---------------------------|----------------------
val_RetrievalMAP           | `todo`
val_RetrievalMRR           | `todo`
val_RetrievalNormalizedDCG | `todo`

### DistilBERT (untrained):
```
./validation.py \
	model=distilbert_dot \
	used_gpus=[1] \
	trainer.devices=1 \
	datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=10 \
	trainer.limit_val_batches=0.1
```

 Metric                    | Score
---------------------------|----------------------
val_RetrievalMAP           | 0.010243686847388744
val_RetrievalMRR           | 0.010256241075694561
val_RetrievalNormalizedDCG | 0.10626860707998276


### DistilBERT (Distilled for 2 epochs from BERT trained for 1 epoch)
```
./validation.py \
	model=distilbert_dot \
	model@teacher=bert_dot \
	checkpoint="./checkpoints/distillbert_mmse_bert_refine/epoch\=1-step\=17698.ckpt" \
	used_gpus=[1] \
	trainer.devices=1 \
	datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=20 \
	trainer.limit_val_batches=0.1
```

 Metric                    | Score
---------------------------|----------------------
val_RetrievalMAP           | 0.03536894544959068
val_RetrievalMRR           | 0.03557067736983299
val_RetrievalNormalizedDCG | 0.14458268880844116


---
ABOVE RESULTS SEEM WRONG the following are redone

---
-->
# DistilBERT (refined) &mdash; `distilbert_refine`
 1. Training:
	```
	./train.py \
		model=distilbert_dot \
		used_gpus=[0,2,3] \
		trainer.devices=3 \
		datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		datamodule.fold_name="fold_0" \
		datamodule.batch_size=10 \
		trainer.limit_val_batches=0 \
		trainer.max_epochs: 20
	```
 2. Run TREC evaluation (after Epoch 8)
	```
	./trec_eval.py \
		model=distilbert_dot \
		model.checkpoint="checkpoints/distilbert_refine/epoch\=8-step\=318573.ckpt" \
		used_gpus=[1] \
		out_path="./trec_evals/distilbert_refined_epoch8.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		batch_size=50
	```
	Results:
	```
		map           0.000013
		recip_rank    0.000016
		ndcg          0.999973
	```
 3. Run TREC evaluation (after Epoch 10)
	```
	./trec_eval.py \
		model=distilbert_dot \
		model.checkpoint="checkpoints/distilbert_refine/epoch\=10-step\=389367.ckpt" \
		used_gpus=[1] \
		out_path="./trec_evals/distilbert_refined_epoch10.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		batch_size=50
	```
	Results:
	```
		map           0.000016
		recip_rank    0.000026
		ndcg          0.999976
	```