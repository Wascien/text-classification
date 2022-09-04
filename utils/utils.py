import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from torch.utils.data import  DataLoader,Dataset
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

#输入训练集，分词器，词表数，最小频率数返回建立的词表
def build_vocab(filePath, tokenizer, maxSize, minFreq):
    vocabDic={}
    with open(filePath,'r',encoding='UTF-8') as f:
        for sentence in tqdm(f):

            sentence=sentence.strip()
            if not sentence:
                break
            content=sentence.split('\t')[0]

            for word in tokenizer(content):

                vocabDic[word]=vocabDic.get(word,0)+1
        vocabList=sorted([item for item in vocabDic.items() if item[1]>=minFreq],key=lambda x:x[1],reverse=True)[:maxSize]

        vocabDic={wordCount[0]:idx for idx , wordCount in enumerate(vocabList) }
        vocabDic.update({UNK:len(vocabDic),PAD:len(vocabDic)+1})

        return vocabDic


def build_dataset(config):
    if config.SplitLevel=='word':##英文可使用
        tokenizer=lambda x:x.split(' ')
    elif config.SplitLevel=='char':##中文常用
        tokenizer=lambda x:[p for p in x]


    if os.path.exists(config.VocabPath):
        vocab=pkl.load(open(config.VocabPath,'rb'))
    else:
        vocab=build_vocab(config.TrainPath,tokenizer=tokenizer,maxSize=MAX_VOCAB_SIZE,minFreq=1)

        pkl.dump(vocab,open(config.VocabPath,'wb'))

    config.NVocab=len(vocab)

    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path,padSize=32):
        contents=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in tqdm(f):
                line=line.strip()
                if not line:
                    continue
                content,label=line.split('\t')
                wordsLine=[]
                token=tokenizer(content)
                seqLen=len(token)
                if len(token)<padSize:
                    token.extend([PAD]*(padSize-len(token)))
                else:
                    token=token[:padSize]
                for word in token:
                    wordsLine.append(vocab.get(word,vocab.get(UNK)))
                contents.append({
                    'text':wordsLine,
                    'label':int(label),
                    'seqlen':seqLen
                })

        return contents

    trainData=load_dataset(config.TrainPath,config.PadSize)
    devData=load_dataset(config.DevPath,config.PadSize)
    testData=load_dataset(config.TestPath,config.PadSize)

    return vocab,trainData,devData,testData


class DataIterGenerater(object):
    class Data(Dataset):
        def __init__(self,data):
            self.data=data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)


    def __init__(self):

        def collate_fn(batchData):
            seqLen,text,labels=[],[],[]
            for sentence in batchData:
                text.append(sentence['text'])
                labels.append(sentence['label'])
                seqLen.append(sentence['seqlen'])
            text=torch.tensor(text).to(self.config.Device)
            labels=torch.tensor(labels).to(self.config.Device)
            seqLen=torch.tensor(seqLen)
            return (text,seqLen),labels

        self.collate_fn=collate_fn


    def create_full_dataiter(self,dataSet):
        config=dataSet.config
        self.config=config
        trainIter=DataLoader(self.Data(dataSet.trainData),batch_size=config.BatchSize,shuffle=config.Shuffle,collate_fn=self.collate_fn)
        devIter = DataLoader(self.Data(dataSet.devData), batch_size=config.BatchSize, shuffle=config.Shuffle,
                               collate_fn=self.collate_fn)
        testIter = DataLoader(self.Data(dataSet.testData), batch_size=config.BatchSize, shuffle=config.Shuffle,
                               collate_fn=self.collate_fn)
        return trainIter,devIter,testIter

    def create_dataiter(self,data,config):
        return DataLoader(self.Data(data),batch_size=config.BatchSize,shuffle=config.Shuffle,collate_fn=self.collate_fn)



class DataSet(object):

    def __init__(self,config):
       self.config=config
       self.vocab,self.trainData,self.devData,self.testData=build_dataset(config)

    def get_traindata(self):
        return self.trainData

    def get_devdata(self):
        return self.devData

    def get_testdata(self):
        return  self.testData

    def get_vocab(self):
        return self.vocab



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

