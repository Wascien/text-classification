import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Config():
    def __init__(self,DataPath,embedding=None):
        self.ModelName='FastText'
        self.TrainPath=os.path.join(DataPath,f'{self.ModelName}Data/train.txt')
        self.DevPath = os.path.join(DataPath, f'{self.ModelName}Data/dev.txt')
        self.TestPath = os.path.join(DataPath, f'{self.ModelName}Data/test.txt')
        self.Classes=[x.strip() for x in open(os.path.join(DataPath,f'{self.ModelName}Data/class.txt'),'r',encoding='utf-8').readlines()]
        self.VocabPath=os.path.join(DataPath,f'{self.ModelName}Data/vocab.pkl')
        self.SavePath=os.path.join(DataPath,f"save_dir/{self.ModelName}.bin")
        self.LogPath=os.path.join(DataPath,f'log/{self.ModelName}')
        self.EmbeddingPretrained=torch.tensor(
            np.load(os.path.join(DataPath,f'{self.ModelName}Data/{embedding}'))['embeddings'].astype('float')
        )if embedding!=None else None
        self.Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.Dropout=0.5
        self.RequireImprovement=1000
        self.ClassNum=len(self.Classes)
        self.EpochsNum=20
        self.BatchSize=128
        self.PadSize=32
        self.LR=1e-3
        self.EmbeddingSize=self.EmbeddingPretrained.size(1)\
            if self.EmbeddingPretrained is not None else 300
        self.HiddenSize=256
        self.NGramVocab=250499
        self.SplitLevel='char' ##默认word
        self.Shuffle=True
        self.NVocab=0
        self.FilterSizes=(2,3,4)
        self.NumFilters=256


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        if config.EmbeddingPretrained is not None:
            self.embedding=nn.Embedding(config.EmbeddingPretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.NVocab,config.EmbeddingSize,padding_idx=config.NVocab-1)

        self.convs=nn.ModuleList(
            [
                nn.Conv2d(1,config.NumFilters,(kernel,config.Embeddingsize))
                for kernel in config.FilterSizes
            ]
        )
        self.dropout=nn.Dropout()
        self.fc=nn.Linear(config.NumFilters*(len(config.FilterSizes)),config.ClassNum)


    def conv_and_pool(self,conv,X):
        Y=F.relu(conv(X)).squeeze(3)
        Y=F.max_pool1d(Y,Y.size(2)).squeeze(2)
        return Y


    def forward(self,X):
        X=self.embedding(X)
        X=X.unsqueeze(1)
        out=torch.cat([self.conv_and_pool(conv,X) for conv in self.convs],1)
        out=self.dropout(out)
        out=self.fc(out)
        return out









