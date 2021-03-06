#!/usr/bin/env bash
# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


ls -d $PWD/eval_img2txt/* > eval_img2txt_list.list_csv
ls -d $PWD/eval_txt2img/* > eval_txt2img_list.list_csv

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

python pretrain_main.py \
  --workerGPU=1 \
  --type=txt2img  \
  --mode=predict \
  --predict_input_fp=eval_txt2img_list.list_csv  \
  --predict_batch_size=64  \
  --output_dir=./fashionbert_out  \
  --pretrain_model_name_or_path=pai-imagebert-base-en \
  --predict_checkpoint_path=./fashionbert_pretrain_model_fin/model.ckpt-54198  \
  --image_feature_size=131072  \
  --input_schema="image_feature:float:131072,image_mask:int:64,input_ids:int:64,input_mask:int:64,segment_ids:int:64,nx_sent_labels:int:1,prod_desc:str:1,text_prod_id:str:1,image_prod_id:str:1,prod_img_id:str:1"  \
