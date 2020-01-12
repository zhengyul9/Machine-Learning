import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as data
import torch
import torch.utils
from PIL import Image 
standardlabels=np.eye(9,dtype=int)

def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)
def save_pkl(fname,obj):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)
def loaddata(path='train_data.pkl'):
    datas=load_pkl(path)
    outdatas=[]
    for data in datas: 
        #print(data.shape)
        data=np.asarray(data,dtype=float)
        img=Image.fromarray(data)
        outdata=torchvision.transforms.ToTensor()(torchvision.transforms.functional.resize(img,(32,32)))
        outdatas.append(outdata)   
    train_data=torch.stack([outdata for outdata in outdatas])
    return train_data
def load_easy_data_label(datapath='train_data.pkl',labelpath='finalLabelsTrain.npy'):
    datas=load_pkl(datapath)
    labels=np.load(labelpath)
    k=0
    outdatas=[]
    outlabels=[]
    for data in datas: 
        #print(data.shape)
        if labels[k]!=1 and labels[k]!=2:
            k=k+1
        else:
            data=np.asarray(data,dtype=float)
            img=Image.fromarray(data)
            outdata=torchvision.transforms.ToTensor()(torchvision.transforms.functional.resize(img,(32,32)))
            outdatas.append(outdata) 
            outlabels.append(labels[k])
            k=k+1 
    outlabels=np.asarray(outlabels)
    train_data=torch.stack([outdata for outdata in outdatas])   
    train_label=torch.tensor(outlabels,dtype=torch.int64)
    print(train_label)
    return train_data,train_label
    
def loadlabel(path='finalLabelsTrain.npy'):
    labels=np.load(path)    
    print(type(labels))     
    trainlabel=torch.tensor(labels,dtype=torch.int64)
    print(trainlabel)
    return trainlabel
def My_Data_Set(datapath='train_data.pkl',labelpath='finalLabelsTrain.npy'):
    train_data=loaddata(datapath)
    train_label=loadlabel(labelpath)
    mydataset=torch.utils.data.TensorDataset(train_data,train_label)
    return mydataset
def My_Data_TestSet(datapath):
    test_data=loaddata(datapath)
    mytestset=torch.utils.data.TensorDataset(test_data)
    return mytestset
def My_Easy_Data_Set(datapath='train_data.pkl',labelpath='finalLabelsTrain.npy'):
    train_data,train_label=load_easy_data_label(datapath,labelpath)
    mydataset=torch.utils.data.TensorDataset(train_data,train_label)
    return mydataset
print(My_Data_Set())
print(My_Easy_Data_Set())
