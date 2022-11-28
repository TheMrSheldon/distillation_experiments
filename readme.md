Check what GPUs are currently unused via `nvidia-smi`.
Let X be an available GPU.

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
		trainer.max_epochs=20
	```
 1. Run TREC evaluation (after Epoch 8)
	```
	./trec_eval.py \
		model=distilbert_dot \
		model.checkpoint="checkpoints/distilbert_refine/epoch\=8-step\=318573.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/distilbert_refined_epoch8.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.3528700866835564, 'recip_rank': 0.7973791066814322, 'ndcg_cut_10': 0.5292193529124727}
	```
 1. Run TREC evaluation (after Epoch 10)
	```
	./trec_eval.py \
		model=distilbert_dot \
		model.checkpoint="checkpoints/distilbert_refine/epoch\=10-step\=389367.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/distilbert_refined_epoch10.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	
	~/trec_eval /home/aanand/trec_evaluation/qrels/2019qrels-pass.txt ./trec_evals/distilbert_refined_epoch10.tsv
	```
	Results:
	```
	{'map': 0.3741730984165488, 'recip_rank': 0.8624031007751939, 'ndcg_cut_10': 0.575479144061756}
	```

# BERT (refined) &mdash; `bert_refine`
 1. Training
	```
	???
	```
 2. Run TREC evaluation (after Epoch 0)
	```
	./trec_eval.py \
		model=bert_dot \
		model.checkpoint="checkpoints/bert_refine/epoch\=0-step\=44247.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_refined_epoch0.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.4446773271758349, 'recip_rank': 0.9180509413067552, 'ndcg_cut_10': 0.6794196539531022}
	```