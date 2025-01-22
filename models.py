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
from models import GNNImageClassificator

warnings.filterwarnings('ignore')



''' Here I have modified the architecture of the GNN by using 4 GATConv and 3 Linear layers'''

class GNNImageClassificator(torch.nn.Module):
    """
    A modified version of the GNNImageClassificator architecture that uses 
    4 GATConv layers and 3 fully connected (FC) layers.
    """
    def __init__(self, in_channels = 3, hidden_dim = 152, num_classes = 10):
        
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_dim)
        self.conv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.conv3 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.conv4 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.fc1 = torch.nn.Linear(in_channels + 4 * hidden_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)
        

    def forward_one_base(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        
        assert node_features.ndim == 2 and node_features.shape[1] == self.in_channels
        assert edge_indices.ndim == 2 and edge_indices.shape[0] == 2
        x0 = node_features
        x1 = self.conv1(x0, edge_indices)
        x2 = self.conv2(x1, edge_indices)
        x3 = self.conv3(x2, edge_indices)
        x4 = self.conv4(x3, edge_indices)
        x0_x1_x2_x3_x4 = torch.cat((x0, x1, x2, x3, x4), dim=-1)
        return x0_x1_x2_x3_x4
    

    def forward(self, batch_node_features: list[torch.Tensor], batch_edge_indices: list[torch.Tensor]) -> torch.Tensor:
        
        assert len(batch_node_features) == len(batch_edge_indices)

        features_list = []
        for node_features, edge_indices in zip(batch_node_features, batch_edge_indices):
            features_list.append(self.forward_one_base(node_features=node_features, edge_indices=edge_indices))

        features = torch.stack(features_list, dim=0) 
        features = features.mean(dim=1) 
        logits = self.fc3(self.fc2(self.fc1(features)))
        return logits



class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.fc = torch.nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN()
print(model)