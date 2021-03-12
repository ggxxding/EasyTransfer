# coding=utf-8
import json
import numpy as np
from transformers.tokenization_bert import BertTokenizer
import random
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
jsonlist_train=[]
jsonlist_dev=[]
jsonlist_eval=[]
'''input_features:
image_feature:float:131072,
image_mask:int:64,
masked_patch_positions:int:5,
input_ids:int:64,
input_mask:int:64,
segment_ids:int:64,
masked_lm_positions:int:10,
masked_lm_ids:int:10,
masked_lm_weights:float:10,
nx_sent_labels:int:1
'''
#3730  6157 empty
with open('wnqdata.txt','r') as f:
    for line in f:
        data = json.loads(line)
        #{"sentids": [8380], "text": "文本", "image": "4190.png", "split": "train", "label": "false"}
        #imgs/xxx.png
        id = int(data['sentids'][0])
        print(id)
        #6704 7542
        text = tokenizer.tokenize(data['text'])
        ids = tokenizer.convert_tokens_to_ids(text)
        if id<=6704:
            print(id)
        elif id>6704 and id<=7542:
            print(id)
        elif id>7542:
            img_feature = np.load('./imgfeatures/' + str(data['image']) + '.npy')
            input = np.array(img_feature, dtype=str).tolist()
            input = ','.join(input)
            input += '\t'
            image_mask = np.ones(64, dtype=str).tolist()
            input += ','.join(image_mask)
            input += '\t'
            # msked_patch_positions
            '''positions = []
            while len(positions) < 5:
                temp = random.randint(1, 63)
                if temp not in positions:
                    positions.append(temp)
            positions.sort()
            masked_patch_position = np.array(positions, dtype=str).tolist()
            input += ','.join(masked_patch_position)
            input += '\t'
            '''
            # input_ids

            # text = tokenizer.tokenize(data['text'])
            # ids = tokenizer.convert_tokens_to_ids(text)
            if len(ids) > 64:
                len_ids = 64
                ids = ids[:64]
            else:
                len_ids = len(ids)
                while (len(ids) != 64):
                    ids.append('0')
            input_ids = np.array(ids, dtype=str).tolist()
            ###input += ','.join(input_ids)
            # input += '\t'
            # input_mask
            input_mask = np.ones(len_ids, dtype=str).tolist()
            while len(input_mask) != 64:
                input_mask.append('0')
            ###input += ','.join(input_mask)
            # input += '\t'
            # segment_ids
            segment_ids = np.zeros(64, dtype=int).tolist()
            for i in range(len(segment_ids)):
                segment_ids[i] = str(segment_ids[i])
            ###input += ','.join(segment_ids)
            # input += '\t'
            # print(segment_ids)
            # print(','.join(segment_ids))
            # masked_lm_positions:int:10
            positions = []
            position_len = round(len_ids * 0.15)
            while len(positions) < position_len:
                temp = random.randint(1, len_ids - 1)
                if temp not in positions:
                    positions.append(temp)
            positions.sort()
            while len(positions) != 10:
                positions.append(0)
            masked_lm_positions = np.array(positions, dtype=str).tolist()
            ###input += ','.join(masked_lm_positions)
            # input += '\t'
            # print(','.join(masked_lm_positions))
            # masked_lm_ids:int:10,masked_lm_weights:float:10,nx_sent_labels:int:1
            masked_lm_ids = []
            for i in masked_lm_positions:
                if len(masked_lm_ids) < position_len:
                    masked_lm_ids.append(str(input_ids[int(i)]))
                else:
                    masked_lm_ids.append(str(0))
            input += ','.join(input_ids)
            input += '\t'
            input += ','.join(input_mask)
            input += '\t'
            input += ','.join(segment_ids)
            input += '\t'
            #input += ','.join(masked_lm_positions)
            #input += '\t'
            #input += ','.join(masked_lm_ids)
            #input += '\t'
            # masked_lm_weights:float:10,
            #masked_lm_weights = np.array(np.ones(10, dtype=float), dtype=str).tolist()
            '''for i in range(position_len, 10):  # range(9,10) -> 9
                masked_lm_weights[i] = '0.0'
            input += ','.join(masked_lm_weights)
            input += '\t'
            '''
            # nx_sent_labels:int:1
            if data['label'] == 'true':
                input += '0'
            else:
                input += '1'
            input += '\t'
            #prod_desc:str:1,
            input += '0\t'
            #text_prod_id:str:1,
            input += str(id)
            input += '\t'
            #image_prod_id:str:1,
            if data['label'] == 'false':
                input += str(id - 4190)
            else:
                input += str(id)
            input += '\t'
            print(id,id-4190)
            #prod_img_id:str:1
            input += '0'


            jsonlist_eval.append(input)
        #print(data)
random.shuffle(jsonlist_eval)

with open('eval_list1.list_csv','w') as f:
    for i in jsonlist_eval:
        f.write(i)
        f.write('\n')
