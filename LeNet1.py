#train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
#train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
#t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
#t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
#http://yann.lecun.com/exdb/mnist/
#from torch import*
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time
import Data
import math

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)
'''def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y'''


def cal_rate(result, thres):#计算一个阈值下的ROC点
    all_number = len(result[0])
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        disease = result[0][item]
        if disease >= thres:
            disease = 1
        if disease == 1:
            if result[1][item] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[1][item] == 0:
                TN += 1
            else:
                FN += 1
    # print TP+FP+TN+FN
    accracy = float(TP+FP) / float(all_number)
    if TP+FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP+FP)
    TPR = float(TP) / float(TP+FN)
    TNR = float(TN) / float(FP+TN)
    FNR = float(FN) / float(TP+FN)
    FPR = float(FP) / float(FP+TN)
    # print accracy, precision, TPR, TNR, FNR, FPR
    return accracy, precision, TPR, TNR, FNR, FPR

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.fc1 = nn.Sequential(
            nn.Linear(25 * 16, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
        )

    def forward(self, x) ->torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        #print('size', x.size())
        #全连接
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def score(self,input):
        with torch.no_grad():
            self.eval()
            correct = 0
            len = 0
            for x, y in input:
                #len += y.shape
                preds = self.forward(x)
                for i, pred in enumerate(preds):
                    if pred.argmax() == y[i]:
                        correct += 1
                #print(correct)
        #print(len)
        print("总正确")
        print(correct)
        return correct/10000


''' def score(self, input, target):
        with torch.autograd.profiler.profile() as prof:
            preds = self.forward(input)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        correct = 0
        for i, pred in enumerate(preds):
            if pred.argmax() == target[i]:
                correct += 1
        return correct/len(input)'''


def fit(epochs, model, loss_func, opt, train_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            #pred = nn.Softmax(model(xb), 1)
            #print(pred.size(), yb.size())
            #print(pred)
            loss = loss_func(pred, yb)
            #print(loss)
            loss.backward()
            opt.step()

        #model.eval()
        print("训练第%d" % epoch + "轮结束")
    return pred


if __name__ == '__main__':
    bs = 128  # batch size
    lr = 0.01  # learning rate
    epochs = 1  # how many epochs to train for
    dev = torch.device(  # 使用GPU
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 训练图像集，训练标记集，训练预测结果集； 测试图像集，测试标记集，测试预测结果集
    train_images, train_labels, train_pres, test_images, test_labels, test_pres = Data.get_data()
    train_images, train_labels, test_images, test_labels = map(  # 映射成torch的张量
        torch.tensor, (train_images, train_labels, test_images, test_labels)
    )

    train_images = train_images.reshape((60000, 1, 28, 28))
    test_images = test_images.reshape((10000, 1, 28, 28))

    train_images = train_images.float()
    train_labels = train_labels.long()

    test_images = test_images.float()
    test_labels = test_labels.long()

    train_images.to(dev)
    train_labels.to(dev)

    train_ds = TensorDataset(train_images, train_labels)
    test_ds = TensorDataset(test_images, test_labels)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)
    print(train_images.shape)
    print(train_labels.shape)
    print("数据加载成功")
    #开始训练
    torch.cuda.synchronize()  # 增加同步操作
    start = time.time()  # 程序起始时间

    model = LeNet()
    print(model.parameters())
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.to(dev)
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    train_pres = fit(epochs, model, loss_func, opt, train_dl)
    #计算评分
    print("训练结束")
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    print("train_score:", model.score(test_dl))








# 计算ROC图  train_labels, train_pres, test_labels, test_pres
    '''prob = train_pres
    label = train_labels
    threshold_vaule = sorted(prob)
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
    # calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob, label), threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR
    # 画出ROC
    AUC = np.trapz(TPR_array, FPR_array)
    threshold = np.argmin(abs(FNR_array - FPR_array))
    EER = (FNR_array[threshold] + FPR_array[threshold]) / 2
    print('AUC : %f' % (-AUC))
    plt.plot(FPR_array, TPR_array)
    plt.title('roc')
    plt.xlabel('FPR_array')
    plt.ylabel('TPR_array')
    plt.show()'''

    torch.cuda.synchronize()  # 增加同步操作
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

