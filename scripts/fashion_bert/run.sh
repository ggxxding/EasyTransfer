
export CUDA_VISIBLE_DEVICES="0"

python pretrain_main.py \
  --workerGPU=1 \
  --type=img2txt  \
  --mode=predict \
  --predict_input_fp=eval_img2txt_list.list_csv  \
  --predict_batch_size=64  \
  --output_dir=./fashionbert_out  \
  --pretrain_model_name_or_path=pai-imagebert-base-en \
  --image_feature_size=131072  \
  --predict_checkpoint_path=./fashionbert_pretrain_model_fin/model.ckpt-54198  \
  --input_schema="image_feature:float:131072,image_mask:int:64,input_ids:int:64,input_mask:int:64,segment_ids:int:64,nx_sent_labels:int:1,prod_desc:str:1,text_prod_id:str:1,image_prod_id:str:1,prod_img_id:str:1"  \
