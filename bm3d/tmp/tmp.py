import pickle

with open('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data_zzy/train/noisy_npy/63_sigma_2.npy','rb') as f:
    data:dict=pickle.load(f)

for k in data.keys():
    print(k,data[k].shape)