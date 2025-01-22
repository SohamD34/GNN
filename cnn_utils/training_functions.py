import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import sys
import pickle
import skimage
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels
from torch_geometric import nn
from torch_geometric.nn import GATConv
import cv2
import click
import tqdm
import warnings
from typing import Callable, cast
import time
from sklearn.model_selection import GridSearchCV, ParameterGrid
from torch_geometric.nn import GATConv
from models import GNNImageClassificator
from gnn_utils.plotting_functions import visualize


warnings.filterwarnings('ignore')




def train_CNN(model, trainloader, device, criterion, optimizer, num_epochs, testloader):
  
    model.to(device)
    start_time = time.time()
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        model.eval() # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print('Validation accuracy after epoch %d: %.2f%%' % (epoch + 1, val_accuracy))
    
        model.train()
        train_losses.append(running_loss / len(trainloader))
        end_time = time.time()
        print('Time taken for training: {:.2f} seconds'.format(end_time - start_time))

    return train_losses, val_accuracies




def test_CNN(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))