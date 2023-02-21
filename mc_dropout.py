import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import seaborn as sns
# %matplotlib inline

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import pymc3 as pm


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0 - p)
        else:
            self.multiplier_ = 0.0

    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input

        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected_ = torch.Tensor(input.shape).uniform_(0, 1) > self.p

        # To support both CPU and GPU.
        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)

        # Multiply output by multiplier as described in the paper [1]
        return torch.mul(selected_, input) * self.multiplier_


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='data/', train=False, transform=transform)

# Visualize 10 image samples in MNIST dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
dataiter = iter(trainloader)
images, labels = next(dataiter)

# plot 10 sample images
_, ax = plt.subplots(1, 10)
ax = ax.flatten()
iml = images[0].numpy().shape[1]
[ax[i].imshow(np.transpose(images[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
[ax[i].set_axis_off() for i in range(10)]
plt.show()
print('label:', labels[:10].numpy())
print('image data shape:', images[0].numpy().shape)


class MLP(nn.Module):
    def __init__(self, hidden_layers=[800, 800], droprates=[0, 0]):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("dropout0", MyDropout(p=droprates[0]))
        self.model.add_module("input", nn.Linear(28 * 28, hidden_layers[0]))
        self.model.add_module("tanh", nn.Tanh())

        # Add hidden layers
        for i, d in enumerate(hidden_layers[:-1]):
            self.model.add_module("dropout_hidden" + str(i + 1), MyDropout(p=droprates[1]))
            self.model.add_module("hidden" + str(i + 1), nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.model.add_module("tanh_hidden" + str(i + 1), nn.Tanh())
        self.model.add_module("final", nn.Linear(hidden_layers[-1], 10))

    def forward(self, x):
        # Turn to 1D
        x = x.view(x.shape[0], 28 * 28)
        x = self.model(x)
        return x


class MLPClassifier:
    def __init__(self, hidden_layers=[800, 800], droprates=[0., 0.], batch_size=128, max_epoch=10, lr=0.1, momentum=0):
        # Wrap MLP model
        self.hidden_layers = hidden_layers
        self.droprates = droprates
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model = MLP(hidden_layers=hidden_layers, droprates=droprates)
        self.model.cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.loss_ = []
        self.test_accuracy = []
        self.test_error = []

    def fit(self, trainset, testset, verbose=True):
        # Training, make sure it's on GPU, otherwise, very slow...
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
        X_test, y_test = next(iter(testloader))
        X_test = X_test.cuda()
        for epoch in range(self.max_epoch):
            running_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.loss_.append(running_loss / len(trainloader))
            if verbose:
                print('Epoch {} loss: {}'.format(epoch + 1, self.loss_[-1]))
            y_test_pred = self.predict(X_test).cpu()
            self.test_accuracy.append(np.mean((y_test == y_test_pred).numpy()))
            self.test_error.append(int(len(testset) * (1 - self.test_accuracy[-1])))
            if verbose:
                print('Test error: {}; test accuracy: {}'.format(self.test_error[-1], self.test_accuracy[-1]))
        return self

    def predict(self, x):
        # Used to keep all test errors after each epoch
        model = self.model.eval()
        outputs = model(Variable(x))
        _, pred = torch.max(outputs.data, 1)
        model = self.model.train()
        return pred

    def __str__(self):
        return 'Hidden layers: {}; dropout rates: {}'.format(self.hidden_layers, self.droprates)


hidden_layers = [800, 800]

### Below is training code, uncomment to train your own model... ###
### Note: You need GPU to run this section ###

# # Define networks
# mlp1 = [MLPClassifier(hidden_layers, droprates=[0, 0], max_epoch=1500),
#         MLPClassifier(hidden_layers, droprates=[0, 0.5], max_epoch=1500),
#         MLPClassifier(hidden_layers, droprates=[0.2, 0.5], max_epoch=1500)]
#
# # Training, set verbose=True to see loss after each epoch.
# [mlp.fit(trainset, testset, verbose=True) for mlp in mlp1]
#
# # Save torch models
# for ind, mlp in enumerate(mlp1):
#     torch.save(mlp.model, 'mnist_mlp1_'+str(ind)+'.pth')
#     # Prepare to save errors
#     mlp.test_error = list(map(str, mlp.test_error))
#
# # Save test errors to plot figures
# open("mlp1_test_errors.txt", "w").write('\n'.join([','.join(mlp.test_error) for mlp in mlp1]))


# Load saved models to CPU
mlp1_models = [torch.load('mnist_mlp1_' + str(ind) + '.pth', map_location={'cuda:0': 'cpu'}) for ind in [0, 1, 2]]

# Load saved test errors to plot figures.
mlp1_test_errors = [error_array.split(',') for error_array in open("mlp1_test_errors.txt", "r").read().split('\n')]
mlp1_test_errors = np.array(mlp1_test_errors, dtype='f')

labels = ['MLP no dropout',
          'MLP 50% dropout in hidden layers',
          'MLP 50% dropout in hidden layers + 20% in input layer']

print(labels)
print(mlp1_test_errors.shape)
plt.figure(figsize=(8, 7))
for i, r in enumerate(mlp1_test_errors):
    plt.plot(range(1, len(r)+1), r, '.-', label=labels[i], alpha=0.6)
plt.ylim([50, 250])
plt.legend(loc=1)
plt.xlabel('Epochs')
plt.ylabel('Number of errors in test set')
plt.title('Test error on MNIST dataset for Multilayer Perceptron')
plt.show()
