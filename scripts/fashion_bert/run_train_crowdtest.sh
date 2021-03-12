ls -d $PWD/crowdtest/train_list.list_csv > train_list.list_csv
ls -d $PWD/crowdtest/dev_list.list_csv > dev_list.list_csv

export CUDA_VISIBLE_DEVICES="0"

python pretrain_main.py \
  --workerGPU=1 \
  --mode=train_and_evaluate \
  --train_input_fp=train_list.list_csv  \
  --eval_input_fp=dev_list.list_csv  \
  --pretrain_model_name_or_path=pai-imagebert-base-zh  \
  --input_sequence_length=64  \
  --train_batch_size=4  \
  --num_epochs=10  \
  --model_dir=./fashionbert_model_dir  \
  --learning_rate=1e-4  \
  --image_feature_size=131072  \
  --input_schema="image_feature:float:131072,image_mask:int:64,masked_patch_positions:int:5,input_ids:int:64,input_mask:int:64,segment_ids:int:64,masked_lm_positions:int:10,masked_lm_ids:int:10,masked_lm_weights:float:10,nx_sent_labels:int:1"  \