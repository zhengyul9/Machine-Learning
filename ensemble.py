import numpy as np
def ensemble(lenet,alexnet,wide_resnet,method='withoutsvm',svm=None,w1=1,w2=1,w3=2):
    results=[]
    if method=='withoutsvm':
        for i in range(len(lenet)):
            result=np.zeros([1,8])
            result[0,lenet[i]-1]+=w1
            result[0,alexnet[i]-1]+=w2
            result[0,wide_resnet[i]-1]+=w3
            max_index=np.argmax(result)
            results.append(max_index+1)
    else:        
        for i in range(len(lenet)):
            result=np.zeros([1,8])
            if svm[i]==-1:
                results.append(-1)
            else :
                result[0,lenet[i]-1]+=w1
                result[0,alexnet[i]-1]+=w2
                result[0,wide_resnet[i]-1]+=w3
                max_index=np.argmax(result)
                results.append(max_index+1)
    return results

        
            

