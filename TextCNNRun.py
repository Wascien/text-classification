from train_eval import train,test
from models.TextCnn import Config,Model
from utils.utils import build_dataset,DataSet,DataIterGenerater
if __name__=='__main__':
    config=Config('D:/myRepositories/myBaselines/data')
    dataset=DataSet(config)
    dataIterGenerater=DataIterGenerater()
    trainIter,devIter,testIter=dataIterGenerater.create_full_dataiter(dataset)
    net=Model(config).to(config.Device)
    train(config,net,trainIter,devIter)