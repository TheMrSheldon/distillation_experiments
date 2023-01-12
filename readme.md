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

# BERT_Dual &mdash; `bert_dual`
 1. Training
	```
	./train.py \
		model=bert_dual \
		used_gpus=[0,1,2,3] \
		trainer.devices=4 \
		datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		datamodule.fold_name="fold_0" \
		datamodule.batch_size=10 \
		trainer.limit_val_batches=0 \
		trainer.max_epochs=20
	```
 1. Run TREC evaluation (after Epoch 6)
	```
	./trec_eval.py \
		model=bert_dual \
		model.checkpoint="checkpoints/bert_dual/epoch\=6-step\=185836.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_epoch6.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.10559151956835361, 'recip_rank': 0.31757201854069556, 'ndcg_cut_10': 0.12235013285052725}
	```
 1. Run TREC evaluation (after Epoch 7)
	```
	./trec_eval.py \
		model=bert_dual \
		model.checkpoint="checkpoints/bert_dual/epoch\=7-step\=212384.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_epoch7.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.10002594264026586, 'recip_rank': 0.22921558162947578, 'ndcg_cut_10': 0.11140008632530818}
	```
 1. Run TREC evaluation (after Epoch 8)
	```
	./trec_eval.py \
		model=bert_dual \
		model.checkpoint="checkpoints/bert_dual/epoch\=8-step\=238932.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_epoch8.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.1001979073785852, 'recip_rank': 0.2942619921782883, 'ndcg_cut_10': 0.12154728640554323}
	```
 1. Run TREC evaluation (after Epoch 9)
	```
	./trec_eval.py \
		model=bert_dual \
		model.checkpoint="checkpoints/bert_dual/epoch\=9-step\=265480.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_epoch9.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.10303797592629996, 'recip_rank': 0.2775895485697927, 'ndcg_cut_10': 0.12889616549057867}
	```

# BERT_Dual v2 &mdash; `bert_dual_v2`
 1. Training
	```
	./train.py \
		model=bert_dual_v2 \
		used_gpus=[0,1,2,3] \
		trainer.devices=4 \
		datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		datamodule.fold_name="fold_0" \
		datamodule.batch_size=10 \
		trainer.limit_val_batches=0 \
		trainer.max_epochs=20
	```
 1. Run TREC evaluation (after Epoch 4)
	```
	./trec_eval.py \
		model=bert_dual_v2 \
		model.checkpoint="checkpoints/bert_dual_v2/epoch\=4-step\=132740.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_v2_epoch4.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.15724754324793216, 'recip_rank': 0.3531139608000032, 'ndcg_cut_10': 0.19680729578460973}
	```
 1. Run TREC evaluation (after Epoch 10)
	```
	./trec_eval.py \
		model=bert_dual_v2 \
		model.checkpoint="checkpoints/bert_dual_v2/epoch\=10-step\=292028.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_v2_epoch10.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.15443748780379177, 'recip_rank': 0.4282327612769776, 'ndcg_cut_10': 0.1979624498740878}
	```
 1. Run TREC evaluation (after Epoch 11)
	```
	./trec_eval.py \
		run_name=bert_dual_v2 \
		model=bert_dual_v2 \
		model.checkpoint="epoch\=11-step\=318576.ckpt" \
		used_gpus=[0] \
		trainer.devices=1 \
		out_path="./trec_evals/bert_dual_v2_epoch11.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.1575069880234407, 'recip_rank': 0.37037518023861704, 'ndcg_cut_10': 0.2158962485811817}
	```

# BERT_Dual v2 - Try 2 &mdash; `bert_dual_v2_2`
 1. Training
	 ```
	./train.py \
		run_name=bert_dual_v2_2 \
		model=bert_dual_v2 \
		used_gpus=[0,1,2,3] \
		trainer.devices=4 \
		datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		datamodule.fold_name="fold_0" \
		datamodule.batch_size=10 \
		trainer.limit_val_batches=0 \
		trainer.max_epochs=20
	```


# GRMM `grmm`
 1. Training
	```
	./train.py \
		run_name=grmm \
		model=grmm \
		used_gpus=[] \
		trainer.devices=1 \
		trainer.accelerator=cpu \
		datamodule.data_dir="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		datamodule.fold_name="fold_0" \
		datamodule.batch_size=10 \
		trainer.limit_val_batches=0 \
		trainer.max_epochs=20
	```
 1. Eval
 	```
	./trec_eval.py \
		run_name=grmm \
		model=grmm \
		model.checkpoint="epochepoch=00.ckpt" \
		used_gpus=[] \
		trainer.devices=1 \
		trainer.accelerator=cpu \
		out_path="./trec_evals/grmm_epoch00.tsv" \
		data_path="/home/tim.hagen/datasets/TRECDL2019Passage/" \
		qrels_path="/home/aanand/trec_evaluation/qrels/2019qrels-pass.txt" \
		batch_size=50
	```
	Results:
	```
	{'map': 0.18441328537471377, 'recip_rank': 0.268255574900559, 'ndcg_cut_10': 0.16102790373869252}
	{'map': 0.15399855609133747, 'recip_rank': 0.26755589313683903, 'ndcg_cut_10': 0.1489037966116541}
	{'map': 0.17382576679955608, 'recip_rank': 0.3309574973472273, 'ndcg_cut_10': 0.19515298732927097}
	```

	Epoch | MAP | MRR | NDCG@10
	------|-----|-----|--------
	0     | 0.184 | 0.268 | 0.161
	1     | 0.154 | 0.268 | 0.149
	2     | 0.174 | 0.331 | 0.195
