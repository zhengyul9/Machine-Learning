import cnn
import loaddata
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import Alexnet as al
import numpy as np
import wideresnet
import ensemble as es
from torch.utils.tensorboard import SummaryWriter
import SVM
writer = SummaryWriter('./')
def train(
    batch_size=20,
    learning_rate=0.003,
    betas=(0.9,0.999),
    alex_learning_rate=0.001
    ):
    #save model
    #return easy train label for lenet and alexnet
    #return hard train label for lenet and alexnet
    #paramters
    '''batch_size=20
    learning_rate=0.003
    betas=(0.9,0.999)
    alex_learning_rate=0.001'''

    #load data and validate 
    data=loaddata.My_Data_Set()
    easy_data=loaddata.My_Easy_Data_Set()
    train_db,val_db=torch.utils.data.random_split(data,[5600,800])
    train_loader=torch.utils.data.DataLoader(train_db,batch_size=batch_size,shuffle=True)
    val_loader=torch.utils.data.DataLoader(val_db,batch_size=batch_size,shuffle=True)
    train_hard_data=torch.utils.data.DataLoader(data)
    train_easy_data=torch.utils.data.DataLoader(easy_data)
    
    ##lenet
    
    net=cnn.Net()
    
    criterion =nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=learning_rate,betas=betas)
    
    
    ##alex
    Alexnet=al.AlexNet()
    Alex_optim=optim.Adam(Alexnet.parameters(),lr=alex_learning_rate,betas=betas)
    ALex_cr=nn.CrossEntropyLoss()

    ## wide resnet
    Wresnet=wideresnet.Wide_ResNet(depth=28,widen_factor=10,dropout_rate=0.5,num_classes=9)
    Wresnet_optim=optim.SGD(Wresnet.parameters(), lr=learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    Wresnet_cr=nn.CrossEntropyLoss()

    ## device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    Alexnet.to(device)
    Wresnet.to(device)
    
    ## train lenet
    for epoch in range(100):
        running_loss=0.0
        test_loss=0.0
        for i,data in enumerate(train_loader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i % 200== 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %f' %
                    (epoch + 1, i + 1, running_loss / 200))
                                    
                writer.add_scalar(' net train/loss',running_loss/ 20,epoch + 1)
                writer.flush()
                running_loss = 0.0
        for i,data in enumerate(val_loader,0):
            with torch.no_grad():
                testinputs,testlabels=data[0].to(device),data[1].to(device)
                net_testoutputs=net(testinputs)
                net_testloss=criterion(net_testoutputs,testlabels)
                test_loss+=net_testloss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] test loss: %f' %
                        (epoch + 1, i + 1, test_loss / 20))
                    
                    writer.add_scalar(' net test/loss',test_loss / 20,epoch + 1)
                    writer.flush()
                    test_loss = 0.0    
    print('Finished Training')
    torch.save(net,'lenet.pkl')

    ## alexnet
    for epoch in range(100):
        running_loss=0.0
        test_loss=0.0
        for i,data in enumerate(train_loader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            Alex_optim.zero_grad()
            Alexnet_outputs=Alexnet(inputs)
            Alexnet_loss=ALex_cr(Alexnet_outputs,labels)
            Alexnet_loss.backward()
            Alex_optim.step()
            running_loss+=Alexnet_loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %f' %
                    (epoch + 1, i + 1, running_loss / 200))                
                writer.add_scalar('alexnet train/loss',running_loss / 200,epoch + 1)
                writer.flush()
                running_loss=0.0
        for i,data in enumerate(val_loader,0):
            with torch.no_grad():
                testinputs,testlabels=data[0].to(device),data[1].to(device)
                Alexnet_testoutputs=Alexnet(testinputs)
                Alexnet_testloss=ALex_cr(Alexnet_testoutputs,testlabels)
                test_loss+=Alexnet_testloss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] test loss: %f' %
                        (epoch + 1, i + 1, test_loss / 20))
                    writer.add_scalar(' alexnet test/loss',test_loss / 20,epoch + 1)
                    writer.flush()
                    test_loss=0.0
    print('Finished Training')




    ##wide resnet
        
    for epoch in range(50):
        running_loss=0.0
        test_loss=0.0
        for i,data in enumerate(train_loader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            Wresnet_optim.zero_grad()
            Wresnet_outputs=Wresnet(inputs)
            Wresnet_loss=Wresnet_cr(Wresnet_outputs,labels)
            Wresnet_loss.backward()
            Wresnet_optim.step()
            running_loss+=Wresnet_loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d]  loss: %f' %
                        (epoch + 1, i + 1, running_loss / 200))
                writer.add_scalar('wide resnet train/loss',running_loss / 200,epoch + 1)
                writer.flush()
                running_loss=0.0

        for i,data in enumerate(val_loader,0):
            with torch.no_grad():
                testinputs,testlabels=data[0].to(device),data[1].to(device)
                Wresnet_testoutputs=Wresnet(testinputs)
                Wresnet_testloss=Wresnet_cr(Wresnet_testoutputs,testlabels)
                test_loss+=Wresnet_testloss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] test loss: %f' %
                        (epoch + 1, i + 1, test_loss / 20))
                    writer.add_scalar('wide resnet test/loss',test_loss / 20,epoch + 1)
                    writer.flush()
                    test_loss = 0.0

    print('Finished Training')
    


    #test lenet

    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %f %%' % (
        100 * correct / total))

    ## test alexnet
    correct = 0
    total = 0
    running_loss=0.0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = Alexnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %f %%' % (
        100 * correct / total))





    ##test wide resnet
    correct = 0
    total = 0
    running_loss=0.0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = Wresnet(images)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %f %%' % (
        100 * correct / total))




    ## save model

    torch.save(Alexnet,'Alex_net.pkl')
    torch.save(net,'lenet.pkl')
    torch.save(Wresnet,'wide_resnet.pkl')




    easy_alexpredicteds=[]
    with torch.no_grad():
        for data in train_easy_data:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = Alexnet(images)
            easy_outputs=outputs.data[:,1:3]
            _,easy_alexpredicted=torch.max(easy_outputs, 1)
            easy_alexpredicted=easy_alexpredicted+1
            easy_alexpredicteds.append(easy_alexpredicted.item())





    easy_netpredicteds=[]
    with torch.no_grad():
        for data in train_easy_data:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = net(images)
            easy_outputs=outputs.data[:,1:3]
            _,easy_netpredicted=torch.max(easy_outputs, 1)
            easy_netpredicteds.append(easy_netpredicted.item()+1)


    Wresnet.eval()
    easy_wrnetpredicteds=[]
    with torch.no_grad():
        for data in train_easy_data:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = Wresnet(images)
            easy_outputs=outputs.data[:,1:3]
            _,easy_wrnetpredicted=torch.max(easy_outputs, 1)
            easy_wrnetpredicteds.append(easy_wrnetpredicted.item()+1)


    #### return hard

    alexpredicteds=[]
    with torch.no_grad():
        for data in train_hard_data:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = Alexnet(images)
            _, alexpredicted = torch.max(outputs.data, 1)
            alexpredicteds.append(alexpredicted.item())





    netpredicteds=[]
    with torch.no_grad():
        for data in train_hard_data:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = net(images)
            _, netpredicted = torch.max(outputs.data, 1)
            netpredicteds.append(netpredicted.item())





    Wresnet.eval()
    wrnetpredicteds=[]
    with torch.no_grad():
        for data in train_hard_data:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = Wresnet(images)
            _, wrnetpredicted = torch.max(outputs.data, 1)
            wrnetpredicteds.append(wrnetpredicted.item())

    easy=es.ensemble(easy_netpredicteds,easy_alexpredicteds,easy_wrnetpredicteds,svm=None,w1=1,w2=1,w3=2)
    hard=es.ensemble(netpredicteds,alexpredicteds,wrnetpredicteds,svm=None,w1=1,w2=1,w3=2)

    return easy,hard

train()
SVM.train_svm()
