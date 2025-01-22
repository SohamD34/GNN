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

warnings.filterwarnings('ignore')



def superpixels_to_2d_image(rec: torch_geometric.data.Data, scale: int = 30, edge_width: int = 1) -> np.ndarray:
    pos = (rec.pos.clone() * scale).int()

    image = np.zeros((scale * 26, scale * 26, 1), dtype=np.uint8)
    for (color, (x, y)) in zip(rec.x, pos):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - scale, y0 - scale

        color = int(float(color + 0.15) * 255)
        color = min(color, 255)

        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)

    for node_ix_0, node_ix_1 in rec.edge_index.T:
        x0, y0 = list(map(int, pos[node_ix_0]))
        x1, y1 = list(map(int, pos[node_ix_1]))

        x0 -= scale // 2
        y0 -= scale // 2
        x1 -= scale // 2
        y1 -= scale // 2

        cv2.line(image, (x0, y0), (x1, y1), 125, edge_width)
    return image




def visualize(dataset, image_name, examples_per_class = 10, classes = tuple(range(10)),figsize = (25, 25),edge_width = 1) :
    
    class_to_examples = {class_ix: [] for class_ix in classes}
    for record in dataset:
        enough = True
        for examples in class_to_examples.values():
            if len(examples) < examples_per_class:
                enough = False
        if enough:
            break
        class_ix = int(record.y)
        if class_ix not in class_to_examples:
            continue
        if len(class_to_examples[class_ix]) == examples_per_class:
            continue
        if len(class_to_examples[class_ix]) > examples_per_class:
            raise RuntimeError
        class_to_examples[class_ix].append(superpixels_to_2d_image(record, edge_width=edge_width))

    plt.figure(figsize=figsize)
    subplot_ix = 1
    for class_ix in classes:
        for example in class_to_examples[class_ix]:
            plt.subplot(len(classes), examples_per_class, subplot_ix)
            subplot_ix += 1
            plt.imshow(example, cmap=plt.cm.binary)
    plt.savefig(image_name)
