import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import time

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))
        return x

def convertTGA(num, color):
    if color:
        img = Image.open("./dataset/color"+str(num)+".tga")
    else:
        img = Image.open("./dataset/depth"+str(num)+".tga")
    # arr = np.asarray(img)
    if color:
        img = img.convert('RGB')
    else:
        img = img.convert('L')
    pixel_data = list(img.getdata())
    if color:
        rgb_array = np.array(pixel_data).reshape(img.size[1], img.size[0], 3)
    else:
        rgb_array = np.array(pixel_data).reshape(img.size[1], img.size[0], 1)
    rgb_array = rgb_array/255.0
    if color:
        rgb_array = np.reshape(rgb_array, (1, 250, 250, 3))
    else:
        rgb_array = np.reshape(rgb_array, (1, 250, 250, 1))
    return rgb_array

def convertColorImages(numImages=4000, numDivisions=40):
    colorDataset = None
    for i in range(numDivisions):
        print(i)
        colorImages = None
        for k in range(int(numImages/numDivisions)):
            imgData = convertTGA(i*numDivisions+k, True)
            if colorImages is not None:
                colorImages = np.concatenate((colorImages, imgData), axis=0)
            else:
                colorImages = imgData
        if colorDataset is not None:
            colorDataset = np.concatenate((colorDataset, colorImages), axis = 0)
        else:
            colorDataset = colorImages
    return colorDataset

def convertDepthImages(numImages=4000, numDivisions=40):
    depthDataset = None
    for i in range(numDivisions):
        print(i)
        depthImages = None
        for k in range(int(numImages/numDivisions)):
            imgData = convertTGA(i*numDivisions+k, False)
            if depthImages is not None:
                depthImages = np.concatenate((depthImages, imgData), axis=0)
            else:
                depthImages = imgData
        if depthDataset is not None:
            depthDataset = np.concatenate((depthDataset, depthImages), axis = 0)
        else:
            depthDataset = depthImages
    return depthDataset

batchSize = 4
datasetSize = 400

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print('Using device: ', device)

print('Converting color dataset...')
colorDataset = convertColorImages(datasetSize, 10)
print('Color dataset shape: ', colorDataset.shape)

print('Converting depth dataset...')
depthDataset = convertDepthImages(datasetSize, 10)
print('Depth dataset shape: ', depthDataset.shape)

# create an instance of the network and pass some data through it
model = MyNet()
if torch.cuda.is_available():
    model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
inputTensor = torch.from_numpy(colorDataset.transpose((0, 3, 1, 2))).float()
targetTensor = torch.from_numpy(depthDataset.transpose((0, 3, 1, 2))).float()
dataset = torch.utils.data.TensorDataset(inputTensor, targetTensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
# c = convertTGA(5, True)
# d = convertTGA(5, False)
# a = np.squeeze(c)
# a = a*255.0
# img = Image.fromarray(np.uint8(a))
# img.show()
# a = np.squeeze(d)
# a = a*255.0
# img = Image.fromarray(np.uint8(a))
# img.show()
for epoch in range(100):
    # res = model(torch.from_numpy(c.transpose((0, 3, 1, 2))).float())
    # a = np.squeeze(res.detach().numpy())
    # a = a*255.0
    # img = Image.fromarray(np.uint8(a))
    # img.show()
    print('Epoch: ', epoch)
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, targets = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        print(inputs.device)
        print(targets.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    scheduler.step()