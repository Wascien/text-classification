import os
import torch
import pickle as pkl
from tqdm import  tqdm
import time
from collections import Counter
from datetime import timedelta
from torch.utils.data import Dataset,DataLoader



MAX_VOCAB_SIZE=10000
UNK,PAD='<UNK>','<PAD>'


def build_vocab(filePath,tokenizer,maxSize,minFreq):

    vocabDict=[]
    with open(filePath,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            line=line.strip()
            if not line:
                continue
            content=line.split('\t')[0]
            for word in tokenizer(content):
                vocabDict.append(word)
        vocabDict=Counter(vocabDict)
        vocabDict=dict(vocabDict)
        vocabList=sorted([a for a in vocabDict.items() if a[1]>=minFreq],key=lambda x:x[1],reverse=True)[:maxSize]
        vocabDict={wordAndCount[0]:idx for idx,wordAndCount in enumerate(vocabList)}
        vocabDict.update({UNK:len(vocabDict),PAD:len(vocabDict)+1})
    return  vocabDict


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


    def bi_gram_hash(sequence,t,buckets):
        if t<=0:
            return 0
        t1 = sequence[t - 1]
        t2 = sequence[t]
        return (t2*14918087*18408749 + t1 * 14918087) % buckets

    def tri_gram_hash(sequence,t,buckets):
        if t<=1:
            return 0
        t1=sequence[t-2]
        t2=sequence[t-1]
        t3=sequence[t]
        return (t3*21788233*14918087*18408749+t2 * 14918087 * 18408749 + t1 * 14918087)% buckets

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


                buckets=config.NGramVocab
                bigram,trigram=[],[]
                for i in range(padSize):
                    aBigram=bi_gram_hash(wordsLine,i,buckets)
                    aTrigram=tri_gram_hash(wordsLine,i,buckets)
                    bigram.append(aBigram)
                    trigram.append(aTrigram)


                contents.append({
                    'text':wordsLine,
                    '2-gram':bigram,
                    '3-gram':trigram,
                    'label':int(label),
                    'seqlen':seqLen
                })


        return contents

    trainData=load_dataset(config.TrainPath,config.PadSize)
    devData=load_dataset(config.DevPath,config.PadSize)
    testData=load_dataset(config.TestPath,config.PadSize)

    return vocab,trainData,devData,testData


class DataIterGenerater(object):
    class FastTextData(Dataset):
        def __init__(self,data):
            self.data=data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)


    def __init__(self):

        def collate_fn(batchData):
            seqLen,text,bigram,trigram,labels=[],[],[],[],[]

            for sentence in batchData:
                text.append(sentence['text'])
                bigram.append(sentence['2-gram'])
                trigram.append(sentence['3-gram'])
                labels.append(sentence['label'])
                seqLen.append(sentence['seqlen'])

            text=torch.tensor(text).to(self.config.Device)
            bigram=torch.tensor(bigram).to(self.config.Device)
            trigram=torch.tensor(trigram).to(self.config.Device)
            labels=torch.tensor(labels).to(self.config.Device)
            seqLen=torch.tensor(seqLen)
            return (text,bigram,trigram,seqLen),labels


        self.collate_fn=collate_fn


    def create_full_dataiter(self,dataSet):
        config=dataSet.config
        self.config=config
        trainIter=DataLoader(self.FastTextData(dataSet.trainData),batch_size=config.BatchSize,shuffle=config.Shuffle,collate_fn=self.collate_fn)
        devIter = DataLoader(self.FastTextData(dataSet.devData), batch_size=config.BatchSize, shuffle=config.Shuffle,
                               collate_fn=self.collate_fn)
        testIter = DataLoader(self.FastTextData(dataSet.testData), batch_size=config.BatchSize, shuffle=config.Shuffle,
                               collate_fn=self.collate_fn)
        return trainIter,devIter,testIter

    def create_dataiter(self,data,config):
        return DataLoader(self.FastTextData(data),batch_size=config.BatchSize,shuffle=config.Shuffle,collate_fn=self.collate_fn)



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


def get_time_dif(startTime):
    endTime = time.time()
    timeDif = endTime - startTime
    return timedelta(seconds=int(round(timeDif)))


if __name__=='__main__':
    pass




















































































