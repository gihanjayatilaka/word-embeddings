import argparse
import re
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
    args.add_argument("--epochs",type=int,default=0)
    args.add_argument("--noLines",type=int,default=50000000000000000)
    args.add_argument("--model",type=str,default="skipgram")#"cbow"
    args.add_argument("--negativeSamples",type=int,default=5)
    args.add_argument("--embeddingDimension",type=int,default=300)
    args.add_argument("--batchSize",type=int,default=32)
    

    args=args.parse_args()


    windowHalfSize=args.windowHalfSize
    embeddingDimension=args.embeddingDimension
    negativeSamples=args.negativeSamples


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
    Y1=[]
    Y0=[]

    for i in range(len(words)):
        
        for j in range(max(0,i-windowHalfSize),min(len(words),i+windowHalfSize+1)):
            if i!=j:
                X.append(i)
                Y1.append(j)
                Y0.append([])
                for k in range(args.negativeSamples):
                    Y0[-1].append(np.random.randint(0,len(words)))
                Y0[-1]=np.array(Y0[-1])                    

    X=np.array(X)
    Y1=np.array(Y1)
    Y0=np.array(Y0)

    print("Finished preparing dataset")
    print("X.shape={} Y1.shape={} Y0.shape={}".format(X.shape,Y1.shape,Y0.shape))
    



    for i in range(10):
        print("---------------",i,"---------------")

        # print(int2wordPrintXY(X[i],Y[i],int2word))

    X=to_categorical(X,num_classes=len(vocab))
    Y1=to_categorical(Y1,num_classes=len(vocab))
    Y0=to_categorical(Y0,num_classes=len(vocab))
    print("X.shape={} Y1.shape={} Y0.shape={}".format(X.shape,Y1.shape,Y0.shape))



    vocabSize=len(vocab)


    batchSize=args.batchSize


    x=tf.placeholder(tf.float32,shape=[batchSize,vocabSize],name="x")
    y0=tf.placeholder(tf.float32,shape=[batchSize,negativeSamples,vocabSize],name="y0")
    y1=tf.placeholder(tf.float32,shape=[batchSize,vocabSize],name="y1")
    
    W=tf.Variable(tf.random_uniform([vocabSize,embeddingDimension],-1.0,1.0),name="W")

    # print("x {}\n y1 {} \n y2 {}\n W {}".format(x,y1,y2,W))
    print(x,y1,y0,W)


    xEmbed=tf.matmul(x,W,name="xEmbed")
    y0Embed=tf.matmul(y0,W,name="y0Embed")
    y1Embed=tf.matmul(y1,W,name="y1Embed")
    

    print(xEmbed,y1Embed,y0Embed,"\n")
    
    loss1=tf.einsum("ij,ij->i",y1Embed,xEmbed,name="loss1")
    loss0=tf.einsum("ijk,ik->ij",y0Embed,xEmbed,name="loss1")
    

    print(loss1,loss0,"\n")

    sigLoss0=tf.nn.sigmoid(loss0,name="sigLoss0")
    sigLoss1=tf.nn.sigmoid(loss1,name="sigLoss1")
    
    print(sigLoss1,sigLoss0,"\n")

    cost=tf.reduce_mean(tf.log(sigLoss0),name="cost_func_negative_samples")
    cost+=tf.reduce_mean(tf.log(sigLoss1),name="cost_func_all_samples")
    cost= tf.identity(cost, name="cost")

    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

    print(cost)
    print(optimizer)

    with tf.Session() as sess:
        batchSize=args.batchSize

        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            for i in range(0,len(X),batchSize):
                xBatch=X[i:i+batchSize]
                y1Batch=Y1[i:i+batchSize]
                y0Batch=Y0[i:i+batchSize]
                _,loss=sess.run([optimizer,cost],feed_dict={x:xBatch,y1:y1Batch,y0:y0Batch})
                print("loss={}".format(loss),"\n")



    
    
    print("End of program")





