import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import time
from utils.utils_fasttext import get_time_dif
from torch.utils.tensorboard import SummaryWriter
def init_network(model,method="xavier",exclude="embedding",seed=123):
    for name,w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method=='xavier':
                    nn.init.xavier_uniform(w)
                elif method=='kaiming':
                    nn.init.kaiming_uniform(w)

                else:
                    nn.init.normal(w)

            elif 'bias' in name:
                nn.init.constant(w,0)



def train(config, model, trainIter, devIter):
    startTime = time.time()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn=nn.CrossEntropyLoss()

    nowBatch = 0  # 记录进行到多少batch
    devBestLoss = float('inf')
    lastImprove = 0  # 记录上次验证集loss下降的batch数

    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.LogPath + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.EpochsNum):
        print('Epoch [{}/{}]'.format(epoch + 1, config.EpochsNum))
        for i, (X, labels) in enumerate(trainIter):
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs,labels)
            loss.sum().backward()

            optimizer.step()

            if nowBatch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs,dim=1).data.cpu()
                trainAcc = metrics.accuracy_score(true, predict)

                devAcc, devLoss = evaluate(config, model, devIter)

                if devLoss < devBestLoss:
                    devBestLoss = devLoss
                    torch.save(model.state_dict(), config.SavePath)
                    improve = '*'
                    lastImprove = nowBatch
                else:
                    improve = ''
                timeDif = get_time_dif(startTime)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Dev Loss: {3:>5.2},  Dev Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(nowBatch, loss.item(), trainAcc, devLoss, devAcc, timeDif, improve))
                writer.add_scalar("loss/train", loss.item(), nowBatch)
                writer.add_scalar("loss/dev", devLoss, nowBatch)
                writer.add_scalar("acc/train", trainAcc, nowBatch)
                writer.add_scalar("acc/dev", devAcc, nowBatch)
                model.train()

            nowBatch += 1
            if nowBatch - lastImprove > config.RequireImprovement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        scheduler.step()  # 学习率衰减

    writer.close()

def test(config, model, testIter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    startTime = time.time()
    testAcc, testLoss, testReport, testConfusion = evaluate(config, model, testIter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(testLoss, testAcc))
    print("Precision, Recall and F1-Score...")
    print(testReport)
    print("Confusion Matrix...")
    print(testConfusion)
    timeDif = get_time_dif(startTime)
    print("Time usage:", timeDif)



def evaluate(config, model, dataIter, test=False):
    model.eval()
    lossTotal = 0
    loss_fn=nn.CrossEntropyLoss()
    predictAll = np.array([], dtype=int)
    labelsAll = np.array([], dtype=int)
    with torch.no_grad():
        for X, labels in dataIter:
            outputs = model(X)
            loss = loss_fn(outputs, labels)
            lossTotal += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs, dim=1).cpu().numpy()
            labelsAll = np.append(labelsAll, labels)
            predictAll = np.append(predictAll, predict)

    acc = metrics.accuracy_score(labelsAll, predictAll)
    if test:
        report = metrics.classification_report(labelsAll, predictAll, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labelsAll, predictAll)
        return acc, lossTotal / len(dataIter), report, confusion
    return acc, lossTotal / len(dataIter)

