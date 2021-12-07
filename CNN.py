import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#mount googledrive
from google.colab import drive
drive.mount('/content/drive')


path = '/content/drive/MyDrive/spectograms_test'

transform = transforms.Compose([transforms.Resize((128,128)), 
                                  transforms.ToTensor()])

#split and make sure everything adds up
dataset = torchvision.datasets.ImageFolder(path, transform=transform)
size = len(dataset)
smalllen = int(0.01*size)
trainlen = int(0.69*size)
vallen = int(0.2*size)
testlen = size - smalllen - trainlen - vallen

small, trains, val, test = torch.utils.data.random_split(dataset, [smalllen, trainlen, vallen, testlen])

print("small size: ", len(small))
print("train size: ", len(trains))
print("val size: ", len(val))
print("test size: ", len(test))

# Prepare Dataloader
batch_size = 20
num_workers = 1

train_loader = torch.utils.data.DataLoader(trains, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

# Visualize some sample data
classes = ['Electronic', 'Folk', 'Hip-Hop', 'Rock']

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display


# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])


#torch.manual_seed(0) # set the random seed



def get_model_name(name, epoch, val_acc):
    path = "FULLmodel_{0}_epoch{1}_valacc{2}".format(name, epoch, val_acc)
    return path

def evaluate(model, data_loader):
    correct = 0
    total = 0
    for imgs, labels in data_loader:        
        if torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


class my_CNN(nn.Module):
        def __init__(self, csize = [64, 128, 256, 512, 1024], ksize = [2, 2, 2, 2], numclasses = 4, inputdim = 3, inputsize = 128):
            super(my_CNN, self).__init__()
            self.name = "CNN"
            self.conv1 = nn.Conv2d(inputdim, csize[0], ksize[0])
            self.conv2 = nn.Conv2d(csize[0], csize[1], ksize[1])
            self.conv3 = nn.Conv2d(csize[1], csize[2], ksize[2])
            self.conv4 = nn.Conv2d(csize[2], csize[3], ksize[3])
            self.pool = nn.MaxPool2d(2, 2)

            self.temp1 = int((inputsize - ksize[0] + 1)/2)
            self.temp2 = int((self.temp1 - ksize[1] + 1)/2)
            self.temp3 = int((self.temp2 - ksize[2] + 1)/2)
            self.temp4 = int((self.temp3 - ksize[3] + 1)/2)
            self.fcin = csize[3]*pow(self.temp4,2)
            self.fc1 = nn.Linear(self.fcin, csize[4])
            self.fc2 = nn.Linear(csize[4], numclasses)

        def forward(self, img):
            x = self.pool(F.relu(self.conv1(img)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = x.view(-1, self.fcin)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
 
 
 """

def overfit(model, train_loader, num_epochs=50, learn_rate = 0.001):
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_acc = []
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):         
            if torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            out = model(imgs)             
            loss = criterion(out, labels) 
            loss.backward()               
            optimizer.step()              
            optimizer.zero_grad()         
            
        train_acc.append(evaluate(model, train_loader))
        print("Epoch: {0}, Accuracy: {1}".format(epoch, train_acc[-1]))
                    
    return train_acc


batch_size = 40

CNN = my_CNN()
print("size: ", sum(p.numel() for p in CNN.parameters() if p.requires_grad))
if torch.cuda.is_available():
  print("using GPU")
  CNN.cuda()

num_workers = 1


small_train = torch.utils.data.DataLoader(small, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

overfit(CNN, small_train)

"""


def train(model, train_dataset, val_dataset, num_epochs=5, learn_rate=0.001, save=True):
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_acc = []
    val_acc = []
    for i in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_images = 0
        for imgs, labels in iter(train_dataset):           
            if torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            out = model(imgs)             
            loss = criterion(out, labels) 
            loss.backward()               
            optimizer.step()              
            optimizer.zero_grad()         
        train_acc.append(evaluate(model, train_dataset))
        val_acc.append(evaluate(model, val_dataset))
        print(("Epoch {}: Train acc: {} |" + "Validation acc: {}").format(i, train_acc[i], val_acc[i]))
        model_path = get_model_name(model.name, i, val_acc[i])
        if (save==True):
          torch.save(model.state_dict(), '/content/drive/MyDrive/APS360_project/saved states' + model_path)
          print(f"Saving {model_path}")
    epochs = np.arange(1, num_epochs + 1)
    return train_acc, val_acc, epochs

num_workers = 1
batch_size = 40
training = torch.utils.data.DataLoader(trains, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)
valid = torch.utils.data.DataLoader(val, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)


from random import randint
#genetic crossover algo

#min and max size of channels
MINSIZE = 64
MAXSIZE = 256

#number of generations used
NUMGEN = 10

#number of channel sizes of the model
NUMCHAN = 5

#number of "species" in the population
NUMPOP = 4

#number of epochs each specie is trained before its score is evaluated
NUMEP = 1

#number of initializations of a specific specie
NUMINT = 3

#save best score and best size in an outer variable
bestScore = 0
bestSize = []

channelSizes = []
for j in range(NUMPOP):
  oneSizes = []
  for k in range(NUMCHAN):
    oneSizes.append(randint(MINSIZE, MAXSIZE))
  channelSizes.append(oneSizes)

largestIdx = -1
secondLargestIdx = -1

import copy

for i in range(NUMGEN):
  print("GEN: ", i)
  print("Channel Sizes: ", channelSizes)
  scores = []
  for j in range(NUMPOP):
    scoresum = []

    #take average of three inits
    for k in range(NUMINT):
      CNN = my_CNN(csize = channelSizes[j])
      print("size: ", sum(p.numel() for p in CNN.parameters() if p.requires_grad))
      if torch.cuda.is_available():
        print("using GPU")
        CNN.cuda()

      
      train_acc, val_acc, epochs = train(CNN, training, valid, num_epochs = NUMEP, save=False)
      scoresum.append(val_acc[NUMEP - 1])
      del CNN

    curScore = sum(scoresum)/NUMINT
    if(curScore > bestScore):
      bestScore = curScore
      bestSize = copy.deepcopy(channelSizes[j])

    scores.append(curScore)

  print("Scores: ", scores)
  largest = 0
  secondLargest = 0

  for idx, score in enumerate(scores):
    if(score > largest):
      secondLargest = largest
      secondLargestIdx = largestIdx
      largest = score
      largestIdx = idx
    elif(score > secondLargest):
      secondLargest = score
      secondLargestIdx = idx

  print("Parents: ", largestIdx, secondLargestIdx)
    
  #crossover
  par1 = copy.deepcopy(channelSizes[largestIdx])
  par2 = copy.deepcopy(channelSizes[secondLargestIdx])
  child1 = []
  child2 = []
  for i in range(len(par1)):
    if(randint(1,2) == 1):
      child1.append(par1[i])
      child2.append(par2[i])
    else:
      child1.append(par2[i])
      child2.append(par1[i])
  
  #mutation
  child1[randint(0, NUMCHAN-1)] = randint(MINSIZE, MAXSIZE)
  child2[randint(0, NUMCHAN-1)] = randint(MINSIZE, MAXSIZE)

  #replacement
  firstRep = -1
  secondRep = -1
  for i in range(len(channelSizes)):
    if(i != largestIdx and i != secondLargestIdx and firstRep == secondRep):
      firstRep = i
    elif(i != largestIdx and i != secondLargestIdx):
      secondRep = i
      break

  channelSizes[firstRep] = copy.deepcopy(child1)
  channelSizes[secondRep] = copy.deepcopy(child2)

#select best model
#train it for many epochs and save it


print("Best sizes: ", bestSize)

FINALEP = 50
CNN = my_CNN(csize = bestSize)
if torch.cuda.is_available():
  CNN.cuda()
train_acc, val_acc, epochs = train(CNN, training, valid, num_epochs = FINALEP)
  



