#6w*32*32，train for 5w,test for 1w.
'''
Description For cifar-10

data_batch_ 1-5----data=10000x(1024+2014+1024)-- uint8s
                   labels=10000x1---(0-9)
batches.meta----label_names--label_names[0] == "airplane"
'''
import math
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
#import torchvision
#from torch import nn
#load_data_fun
import  os

from torch.utils.data import TensorDataset, DataLoader


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar10_plot(data, meta, im_idx=0):
    # Get the image data np.ndarray
    im = data[b'data'][im_idx, :] #'data'下面的第 im_idx行的一行数据是一个图片

    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    # 1-D arrays.shape = (N, ) ----> reshape to (1, N, 1)
    # 2-D arrays.shape = (M, N) ---> reshape to (M, N, 1)
    img = np.dstack((im_r, im_g, im_b))
    # img.shape = (32, 32, 3)
    print(img)
    print("shape: ", img.shape)
    print("label: ", data[b'labels'][im_idx])
    print("category:", meta[b'label_names'][data[b'labels'][im_idx]])

    plt.imshow(img)
    plt.show()

def cifar10_data():
    num_images = 10000
    train_images = np.empty((50000, 3, 32, 32), dtype='uint8')
    test_images = np.empty((10000, 3, 32, 32), dtype='uint8')
    train_labels = np.empty((50000), dtype='uint8')
    test_labels = np.empty((10000), dtype='uint8')
    for k in range (5):
        data = unpickle("D:\\project\\PY\\Lenet\\data\\cifar-10-batches-py\\data_batch_" + str(k+1))
        ims = data[b'data']
        lbs = data[b'labels']
        for i in range(num_images):
                train_images[k*10000+i] = ims[i].reshape(3, 32, 32)#np.dstack((im_r, im_g, im_b))
                train_labels[k*10000+i] = lbs[i]

    test_data = unpickle("D:\\project\\PY\\Lenet\\data\\cifar-10-batches-py\\test_batch")
    ims = test_data[b'data']
    lbs = test_data[b'labels']
    for i in range(num_images):
        test_images[i] = ims[i].reshape(3, 32, 32)  # np.dstack((im_r, im_g, im_b))
        test_labels[i] = lbs[i]

    return train_images,  test_images, train_labels,  test_labels

def plotcifa10(args):
    #trainset = unpickle("D:\project\PY\Lenet\data\cifar-10-batches-py\data_batch_1")

    batch = (args // 10000) + 1
    idx = args - (batch-1)*10000
    data = unpickle("D:\\project\\PY\\Lenet\\data\\cifar-10-batches-py\\data_batch_" + str(batch))
    meta = unpickle("D:\\project\\PY\\Lenet\data\\cifar-10-batches-py\\batches.meta")
    cifar10_plot(data, meta, im_idx=idx)


'''self.avg_pool = nn.AdaptiveAvgPool2d(1)
self.fc = nn.Sequential(
    nn.Linear(channel, channel // reduction, bias=False),
    nn.ReLU(inplace=True),
    nn.Linear(channel // reduction, channel, bias=False),
    nn.Sigmoid()
        def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

)'''
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    #inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class SELayer(nn.Module):#SE层
    def __init__(self):
        super(SELayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )

        #self.pool1 = nn.MaxPool2d((2, 2), 2)
        #self.pool2 = nn.MaxPool2d((2, 2), 2)
        #self.pool3 = nn.MaxPool2d((2, 2), 2)
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 16, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 10),
        )

    def forward(self, x) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        #print('size:', x.size())
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

    def score(self,input,num):
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
        return correct/num

class TestModel(nn.Module):#SE层
    def __init__(self):
        super(TestModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.num_flat_features(x))
        #print('size:', x.size())
        #全连接
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def score(self,input,num):
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
        return correct/num





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    #inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        #self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def score(self,input,num):
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
        return correct/num

def fit(epochs, model, loss_func, lr, train_dl):
    lossarr = []
    for epoch in range(epochs):
        model.train()
        if epoch >20:
            lr = 0.01
        if epoch >30:
            lr = 0.001
        if epoch > 35:
            lr = 0.0001
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            #print(pred.size(), yb.size())
            loss = loss_func(pred, yb)
            #print(loss)
            loss.backward()
            opt.step()

        #model.eval()
        lossarr.append(loss)
        print("训练第%d" % epoch + "轮结束,Loss:", loss)
    return pred, lossarr

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
    return x.view(-1, 3, 32, 32).to(dev), y.to(dev)

if __name__ == '__main__':
    torch.cuda.synchronize()  # 增加同步操作
    start = time.time()  # 程序起始时间
    train_images,  test_images, train_labels,  test_labels = cifar10_data()

    #验证图预处理正确
    #index = 1001
    #img = np.dstack((train_images[index][0], train_images[index][1], train_images[index][2]))
    #plt.imshow(img)
    #plt.show()
    #print(train_labels[index])
    #plotcifa10(index)


    bs = 256  # batch size
    lr = 0.1  # learning rate
    epochs = 40  # how many epochs to train for

    train_images,  test_images, train_labels,  test_labels = map(  # 映射成torch的张量
        torch.tensor, (train_images,  test_images, train_labels,  test_labels))

    train_images = train_images.float()
    train_labels = train_labels.long()

    test_images = test_images.float()
    test_labels = test_labels.long()

    train_ds = TensorDataset(train_images, train_labels)
    test_ds = TensorDataset(test_images, test_labels)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)

    print("数据加载结束")
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = TestModel()
    #model = SELayer()
    #model = ResNet(BasicBlock, [2, 2, 2, 2])

    model.to(dev)
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    train_pres, lossarr = fit(epochs, model, loss_func, lr,  train_dl)


    print("训练结束")
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    print("train_dl_score:", model.score(train_dl, 50000))
    print("test_score:", model.score(test_dl, 10000))
    torch.cuda.synchronize()  # 增加同步操作
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

    plt.plot( lossarr, "ro")  # use pylab to plot x and y
    plt.savefig('loss.jpg')
    #plt.show()  # show the plot on the screen










