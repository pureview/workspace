import math
import numpy as np
import random

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def regression(mat,labels,names=None,learning_rate=0.01,split_rate=0.7,
               loss='cross_entropy',epoch=10,initializer='normal'):
    def cal_acc(data,label,weights,bias):
        before_sigmoid = np.sum(data * weights, axis=1) + bias
        # print('before_sigmoid:',before_sigmoid)
        test_out = 1 / (np.exp(-before_sigmoid) + 1)
        # print('test_out:',test_out)
        acc = np.sum(np.equal(label, np.sign(test_out - 0.5))) / label.shape[0]
        return acc

    split=int(len(mat)*split_rate)
    testset=np.array(mat[split:])
    dataset=np.array(mat[:split])
    train_label=np.array(labels[:split])
    test_label=np.array(labels[split:])
    m,n=dataset.shape
    weights=np.random.normal(scale=0.1,size=n)
    bias=0.
    for i in range(epoch):
        for j in range(m):
            y=np.sum(dataset[j]*weights)+bias
            #print('y:',y)
            pred=1/(math.exp(-y)+1)
            #loss=-train_label[j]*math.log(pred)-(1-train_label[j])*math.log(1-pred)
            gradient_w=(train_label[j]-pred)*dataset[j]
            gradient_b=train_label[j]-pred
            # update params
            weights-=learning_rate*gradient_w
            bias-=learning_rate*gradient_b
        #print('training acc=',cal_acc(dataset,train_label,weights,bias))
        print('test acc=',cal_acc(testset,test_label,weights,bias))
        continue
        # cal accuracy
        #print('weights:',weights,'\n','bias:',bias)
        before_sigmoid=np.sum(testset*weights,axis=1)+bias
        #print('before_sigmoid:',before_sigmoid)
        test_out=1/(np.exp(-before_sigmoid)+1)
        #print('test_out:',test_out)
        acc=np.sum(np.equal(test_label,np.sign(test_out-0.5)))/test_label.shape[0]
        print('epoch - %d, acc - %.2f'%(i,acc))

if __name__ == '__main__':
    mat,label=loadDataSet()
    regression(mat,label,epoch=500)