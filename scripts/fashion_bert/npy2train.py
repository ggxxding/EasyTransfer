import numpy as np
import pandas as pd
import csv
data = np.load('./193603_3.png.npy')

arr=np.array(data,dtype=str).tolist()

arr=','.join(arr)

arr+='\t'
image_mask=np.ones(64,dtype=str).tolist()
arr+=','.join(image_mask)
arr+='\t'
masked_patch_position=np.ones(5,dtype=str).tolist()
arr+=','.join(masked_patch_position)
arr+='\t'
arr+=','.join(image_mask)
arr+='\t'
arr+=','.join(image_mask)
arr+='\t'
arr+=','.join(image_mask)
arr+='\t'
masked_lm_positions=np.ones(10,dtype=str).tolist()
arr+=','.join(masked_lm_positions)
arr+='\t'
arr+=','.join(masked_lm_positions)
arr+='\t'
arr+=','.join(masked_lm_positions)
arr+='\t'
arr+='1'
print(arr)
with open('193603.csv','w') as wf:
	for i in range(1000):
		wf.write(arr+'\n')