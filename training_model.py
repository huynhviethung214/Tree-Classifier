import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn import Conv2d, Linear, BatchNorm2d, Module, ReLU, Sequential, MaxPool2d, CrossEntropyLoss
from torchvision import transforms
from PIL import Image
import numpy as np
from icrawler.builtin import GoogleImageCrawler
import os

def download_dataset(names, path, is_downloaded:bool, num_im:int):
    if is_downloaded:
        for name in names:
            _path = path.format(name)
            if not os.path.exists(_path):
                os.makedirs(_path)
            google_crawler = GoogleImageCrawler(storage={'root_dir': path.format(name)})
            google_crawler.crawl(keyword=name, max_num=100)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'D:\\Python Projects\\Project Alpha\\S&U\\PyTorch\\TreeRecognition\\test_set\\{0}'
nameList = ['Nhãn',
            'Xoài',
            'Cam',
            'Quýt',
            'Ôỉ',
            'Bưởi']

def preprocessing(names, _path):
    dataset = []

    for i, name in enumerate(names):
        for im in os.listdir(_path.format(name)):
            im = Image.open('{0}\\{1}'.format(_path.format(name), im))
            im = im.resize((300, 300))
            im = np.array(im)
            im = torch.from_numpy(im)

            if (im.shape == (300, 300, 3)):
                label = torch.from_numpy(np.array([i]))
                label = label.type(torch.cuda.FloatTensor)
                im = im.type(torch.cuda.FloatTensor)

                dataset.append([im, label])

    return np.array(dataset)


class NN(Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = Sequential(
            Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(
            Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = Sequential(
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = Sequential(
            Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = Sequential(
            Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = Sequential(
            Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.fc = Linear(512, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

model = NN().to(device)

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

download_dataset(nameList, path, False, 600)
dataset = preprocessing(nameList, path)
np.random.shuffle(dataset)

# for epoch in range(20):
#     for i, (im, label) in enumerate(dataset):
#         im = im.view(1, 3, 300, 300)
#         label = label.type(torch.cuda.LongTensor)
#         _in = Variable(im).to(device)
#         label = Variable(label).to(device)

#         output = model(_in)
#         loss = criterion(output, label).to(device)

#         model.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if i % 100 == 0:
#             print('[{0}: {1}]-->[Prediction:{2} | Expected Prediction: {3}]'.format(epoch, loss, torch.argmax(output), label))

# torch.save(model.state_dict(), 'model_weights.prmt')

with torch.no_grad():
    score = 0
    # _im = None

    for epoch in range(20):
        for i, (im, label) in enumerate(dataset):
            # _im = im.cpu().numpy().astype(np.uint8)
            im = im.view(1, 3, 300, 300)
            _in = Variable(im).to(device)
            output = model(_in)
            label = Variable(label).to(device)

            pred = torch.argmax(output)
            # print(type(label) == type(pred))
            _target = label.tolist()[0]
            _pred = pred.item()

            if _target == _pred:
                score += 1

    print('Accuracy: {0}'.format((score / len(dataset)) * 100))
    # plt.imshow(_im)
