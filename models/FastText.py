import torch
from torch import nn
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

class Model(nn.Module):


    def __init__(self, config):

        super(Model, self).__init__()
        if config.EmbeddingPretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.EmbeddingPretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.NVocab, config.EmbeddingSize, padding_idx=config.NVocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.NGramVocab, config.EmbeddingSize)
        self.embedding_ngram3 = nn.Embedding(config.NGramVocab, config.EmbeddingSize)
        self.dropout = nn.Dropout(config.Dropout)
        self.fc1 = nn.Linear(config.EmbeddingSize * 3, config.HiddenSize)

        self.fc2 = nn.Linear(config.HiddenSize, config.ClassNum)


    def forward(self, X):

        out_word = self.embedding(X[0])
        out_bigram = self.embedding_ngram2(X[1])
        out_trigram = self.embedding_ngram3(X[2])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
