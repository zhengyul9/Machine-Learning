import cnn
import loaddata
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import Alexnet as al
import ensemble as es
import wideresnet as wr
import numpy as np
import torchvision as cv
import SVM
easy_test_path='train_data.pkl'
hard_test_path='train_data.pkl'
def test(path1,path2):
    #return easy and hard test result for alexnet and lenet
    ##load data
    easy_test_data=loaddata.My_Data_TestSet(path1)
    hard_test_data=loaddata.My_Data_TestSet(path2)
    svmdata=loaddata.load_pkl(path2)
    #
    #trainlabel=loaddata.loadlabel()
    #datatest
    transform=cv.transforms.Compose(
        [
            cv.transforms.Resize([32,32]),
            cv.transforms.ToTensor()
        ]
    )
    easy_test_loader=torch.utils.data.DataLoader(easy_test_data)
    hard_test_loader=torch.utils.data.DataLoader(hard_test_data)
    Alexnet=torch.load('Alex_net.pkl')
    net=torch.load('lenet.pkl')
    Wresnet=torch.load('wide_resnet.pkl')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    Alexnet.to(device)
    Wresnet.to(device)

    easy_alexpredicteds=[]
    with torch.no_grad():
        for data in easy_test_loader:
            images= data[0].to(device)
            outputs = Alexnet(images)
            easy_outputs=outputs.data[:,1:3]
            _,easy_alexpredicted=torch.max(easy_outputs, 1)
            easy_alexpredicted=easy_alexpredicted+1
            easy_alexpredicteds.append(easy_alexpredicted.item())

    easy_netpredicteds=[]
    with torch.no_grad():
        for data in easy_test_loader:
            images= data[0].to(device)
            outputs = net(images)
            easy_outputs=outputs.data[:,1:3]
            _,easy_netpredicted=torch.max(easy_outputs, 1)
            easy_netpredicteds.append(easy_netpredicted.item()+1)


    Wresnet.eval()
    easy_wrnetpredicteds=[]
    with torch.no_grad():
        for data in easy_test_loader:
            images= data[0].to(device)
            outputs = Wresnet(images)
            easy_outputs=outputs.data[:,1:3]
            _,easy_wrnetpredicted=torch.max(easy_outputs, 1)
            easy_wrnetpredicteds.append(easy_wrnetpredicted.item()+1)

    #### return hard

    alexpredicteds=[]
    with torch.no_grad():
        for data in hard_test_loader:
            images= data[0].to(device)
            outputs = Alexnet(images)
            _, alexpredicted = torch.max(outputs.data, 1)
            alexpredicteds.append(alexpredicted.item())





    netpredicteds=[]
    with torch.no_grad():
        for data in hard_test_loader:
            images= data[0].to(device)
            outputs = net(images)
            _, netpredicted = torch.max(outputs.data, 1)
            netpredicteds.append(netpredicted.item())

    Wresnet.eval()
    wrnetpredicteds=[]
    with torch.no_grad():
        for data in hard_test_loader:
            images= data[0].to(device)
            outputs = Wresnet(images)
            _, wrnetpredicted = torch.max(outputs.data, 1)
            wrnetpredicteds.append(wrnetpredicted.item())
    svm_for_work=SVM.test_svm(svmdata)
    easy=es.ensemble(easy_netpredicteds,easy_alexpredicteds,easy_wrnetpredicteds,svm=None,w1=1,w2=1,w3=2)
    hard=es.ensemble(netpredicteds,alexpredicteds,wrnetpredicteds,svm=svm_for_work,w1=1,w2=1,w3=2)
    '''correct=0
    for i in range(len(hard)):
        if hard[i]==trainlabel[i]:
            correct+=1
        i+=1
    print(correct/i)'''
    print("easy: ", easy)
    print("hard: ", hard)
    return easy,hard 
test(easy_test_path,hard_test_path)
