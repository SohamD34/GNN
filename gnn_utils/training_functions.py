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




def build_mnist_superpixels_dataset(train: bool) -> MNISTSuperpixels:
    return MNISTSuperpixels(
        root="mnist-superpixels-dataset",
        train=train,
    )


def build_collate_fn(device: str | torch.device):
    def collate_fn(original_batch: list[Data]):
        batch_node_features: list[torch.Tensor] = []
        batch_edge_indices: list[torch.Tensor] = []
        classes: list[int] = []

        for data in original_batch:
            node_features = torch.cat((data.x, data.pos), dim=-1).to(device)
            edge_indices = data.edge_index.to(device)
            class_ = int(data.y)

            batch_node_features.append(node_features)
            batch_edge_indices.append(edge_indices)
            classes.append(class_)

        collated = {
            "batch_node_features": batch_node_features,
            "batch_edge_indices": batch_edge_indices,
            "classes": torch.LongTensor(classes).to(device),
        }

        return collated

    return collate_fn


def build_dataloader(
    dataset: MNISTSuperpixels,
    batch_size: int,
    shuffle: bool,
    device: str | torch.device,
) -> DataLoader:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collate_fn(device=device),
    )

    return loader


def build_train_val_dataloaders(batch_size: int, device: str) -> tuple[DataLoader, DataLoader]:
    train_dataset = build_mnist_superpixels_dataset(train=True)
    val_dataset = build_mnist_superpixels_dataset(train=False)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    return train_loader, val_loader



def train_one_epoch(model,optimizer,train_loader,criterion,batches_passed):
    
    model.train()
    running_loss = 0.0
    total_correct = 0
    total = 0
    
    for batch in train_loader:
        batch_node_features = batch["batch_node_features"]
        batch_edge_indices = batch["batch_edge_indices"]
        classes = batch["classes"]

        logits = model(batch_node_features=batch_node_features, batch_edge_indices=batch_edge_indices)
        predicted_classes = torch.argmax(logits, dim=1)

        loss = criterion(logits, classes).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_correct += (predicted_classes == classes).to(torch.float32).sum()
        total += len(batch)
        batches_passed += 1
        
    print("Train accuracy =",total_correct/total, "Train loss =",running_loss/len(train_loader))
    return batches_passed


@torch.no_grad()
def evaluate(model,val_loader,epochs_passed):
    model.eval()

    accuracy_sum: float = 0.0
    num_samples: int = 0

    for batch in val_loader:
        batch_node_features = batch["batch_node_features"]
        batch_edge_indices = batch["batch_edge_indices"]
        classes = batch["classes"]

        logits = model(batch_node_features=batch_node_features, batch_edge_indices=batch_edge_indices)
        predicted_classes = torch.argmax(logits, dim=1)

        accuracy_sum += float((predicted_classes == classes).to(torch.float32).mean().cpu().numpy()) * len(classes)
        num_samples += len(classes)

    accuracy = accuracy_sum / num_samples

    print("Val_accuracy", accuracy)




def train(batch_size, epochs, device, hidden_dim, lr):
    
    model = GNNImageClassificator(in_channels=3, hidden_dim=hidden_dim).to(device)
    train_loader, val_loader = build_train_val_dataloaders(batch_size=batch_size, device=device)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    visualize(
        cast(MNISTSuperpixels, train_loader.dataset),
        image_name="all_classes.jpg",
    )

    visualize(
        cast(MNISTSuperpixels, train_loader.dataset),
        image_name="one_class.jpg",
        classes=(4,),
        examples_per_class=1,
    )

    batches_passed = 0

    for epoch_ix in range(epochs):
        start_time = time.time()
        print("Epoch:",epoch_ix)
        batches_passed = train_one_epoch(model=model, optimizer=optimizer, train_loader=train_loader, criterion=torch.nn.CrossEntropyLoss(), batches_passed=batches_passed)
        evaluate(model=model, val_loader=val_loader, epochs_passed=epoch_ix + 1)
        end_time = time.time()
        print("Time for epoch:",end_time - start_time)


