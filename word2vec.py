import argparse
import re
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Input, Flatten
from tensorflow.keras.utils import to_categorical


def int2wordX(ar, int2wordDict):
    words = []
    
    try:
        for i in ar:
            words.append("{}.{} ".format(i,int2wordDict[i]))
    except:
        words=int2wordDict[ar]
    return ''.join(words)


def int2wordPrintXY(x,y,int2wordDict):
    return "X={} | Y={}".format(int2wordX(x,int2wordDict),int2wordX(y,int2wordDict))


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--dataset",type=str,default="./data/shakespeare.txt")
    args.add_argument("--windowHalfSize",type=int,default=2)
    args.add_argument("--epochs",type=int,default=50)
    args.add_argument("--noLines",type=int,default=50000000000000000)
    args.add_argument("--model",type=str,default="cbow")#"skipgram"

    args=args.parse_args()


    windowHalfSize=args.windowHalfSize
    

    with open(args.dataset) as f: 
        lines = f.read().splitlines()
    
    lines=lines[:args.noLines]


    words=set([])

    for line in lines:
        line = re.split('[^a-zA-Z]+',line.lower())
        line = [word for word in line if len(word)>0]
        for word in line:
            words.add(word)
    
    vocab=sorted(list(words))
    word2int={}
    int2word={}
    int2wordNP=[]

    for i,word in enumerate(vocab):
        word2int[word]=i
        int2word[i]=word
        int2wordNP.append(word)

    int2wordNP=np.array(int2wordNP)

    X=[]
    Y=[]

    for line in lines:
        line = re.split('[^a-zA-Z]+',line.lower())
        line = [word for word in line if len(word)>0]

        # for contextIndex in range(0,len(line)-windowHalfSize*2-1,2*windowHalfSize):
        for contextIndex in range(0,len(line)-windowHalfSize*2-1,1):

            xThis=[]
            for i in range(contextIndex,contextIndex+windowHalfSize*2+1):
                if i != contextIndex+windowHalfSize:
                    xThis.append(word2int[line[i]])
            xThis=np.array(xThis)
            X.append(xThis)
            Y.append(word2int[line[contextIndex+windowHalfSize]])

    X=np.array(X)
    Y=np.array(Y)
    
    print("X.shape={}".format(X.shape))
    print("Y.shape={}".format(Y.shape))
    

    for i in range(10):
        print("---------------",i,"---------------")

        print(int2wordPrintXY(X[i],Y[i],int2word))


    X=to_categorical(X,num_classes=len(vocab))
    Y=to_categorical(Y,num_classes=len(vocab))

    print("X.shape={}".format(X.shape))
    print("Y.shape={}".format(Y.shape))


    if args.model=="cbow":
        y1=Input(shape=(X.shape[1],X.shape[2]))
        y2=Flatten()(y1)
        y3=Dense(100)(y2)
        y4=Dense(Y.shape[1],activation="softmax")(y3)
    elif args.model=="skipgram":
        y1=Input(shape=(X.shape[1]))
        y2=Flatten()(y1)
        y3=Dense(100)(y2)
        y4=Dense((Y.shape[1],Y.shape[2]),activation="softmax")(y3)

    # embed=y2.output

    model=Model(inputs=y1,outputs=y4)
    model.compile(loss="categorical_crossentropy",optimizer="adam")
    model.summary()

    model.fit(X,Y,epochs=args.epochs,batch_size=32)

    print(len(vocab))
    print(vocab[:100])


