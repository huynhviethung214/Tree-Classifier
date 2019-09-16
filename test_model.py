from PIL import Image
import torch
import numpy as np
from torch.nn import Module, Conv2d, Sequential, MaxPool2d, ReLU, BatchNorm2d, Linear
from torch.autograd import Variable
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nameList = ['Nhãn',
            'Xoài',
            'Cam',
            'Quýt',
            'Ôỉ',
            'Bưởi']

f_list = ['Nhan',
          'Xoai',
          'Cam',
          'Quyt',
          'Oi',
          'Buoi']

path = 'D:\\Python Projects\\Project Alpha\\S&U\\PyTorch\\TreeRecognition\\test_set'

def preprocessing(f_name, _index):
    im = Image.open('{0}\\{1}\\{2}.jpg'.format(path, f_name, _index))
    im = im.resize((300, 300))
    im = np.array(im)
    im = torch.from_numpy(im)

    if (im.shape == (300, 300, 3)):
        im = im.type(torch.cuda.FloatTensor)
        return im

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

model = NN().cuda(device)
model.load_state_dict(torch.load('model_weights.prmt'))

score = 0
batch = 80
exc = 0

for name in f_list:
    for i in range(1, batch):
        try:
            im = preprocessing(name, '0000{0}'.format('0{0}'.format(i) if i < 10 else '{0}'.format(i)))

            im = im.view(1, 3, 300, 300)
            _in = Variable(im).to(device)
            output = model(_in)
            pred = torch.argmax(output)
            _pred = pred.item()
            print('Label: ' + nameList[_pred], ' Predicted: ' + name)

            if f_list[_pred] == name:
                score += 1

        except:
            exc -= 1

acc = (score / ((batch * len(nameList)) - exc)) * 100
print(acc)