# project-01-8086

## train  enviroment  

Windows 10 - 64 bit
AMD R5 3600 4.0GHZ + NVDIA RTX 2060 6G + 16GB 2933MHZ*2.

## Packages

In order to run this project, we need to install:
1. Python3.6
2. Anaconda
3. Pytorch
4. PIL
5. torchvision
6. sklearn
7. joblib
8. pickle
9. time

## train.py

Inputs are some parameters we will use in the train. Outputs are lists of predeicted labels which we use our model to compute on the easy and hard train dataset.  

you can change the parameters in train.py by changing input of train() function. Such as :
$ train(    batch\_size=20,
    learning\_rate=0.003,
    betas=(0.9,0.999),
    alex\_learning\_rate=0.001)$

In this python file, we train our three networks and SVM. It may be take a long time, since our model is a bit complicated.

## test.py  

Because the model is loaded in this step, and our wide-resnet parameter file is too large to upload, we compressed it to one zip file. The user should Unzip 'wide\_resnet' first.

Input of this file is the path of easy test data path and hard test data path. Output is two lists of labels on these data.

We can change the data path in test.py by changing $easy\_test\_path='train\_data.pkl'$ 
$hard\_test\_path='train\_data.pkl'$ or you can change the input of test(easy\_test\_path,hard\_tes\_path) to whatever you want to test. 

## how to run

1. Download zip and unzip (or clone) the project file on Github and open 'project-01-8086k' folder.
2. Unzip 'wide\_resnet.zip'.
3. Open 'test.py' and change test file path as 'test.py' section described above.
4. Run 'test.py' 
5. If there is an error while running 'test.py', please open 'train.py' and inilialize parameters as described above, and run "train.py" on your machine first, then run 'test.py')

 