import os
import time
import logging

import pandas as pd
import torch
import numpy as np
import torchvision.models as models
from torch import nn
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

from crawler.admin import check_folder_mk, hash_file_name


# Fixed AnnotationsLoader to properly append _dfs and fix mean size calculation
class AnnotationsLoader:
    """
    Loads and combines image annotation CSV files for datasets.

    Attributes:
        _img_labels (pd.DataFrame): Combined DataFrame of all annotations.
        _mean_height (int): Mean height of images in dataset.
        _mean_width (int): Mean width of images in dataset.
        _dfs (list): List of DataFrames loaded from multiple CSVs.

    Args:
        annotations_file (str or list): Path(s) to annotation CSV file(s).
    """
    _img_labels: pd.DataFrame
    _mean_height: int
    _mean_width: int
    _dfs: list

    def __init__(self, annotations_file: str | list):
        if isinstance(annotations_file, str):
            self._img_labels = pd.read_csv(annotations_file)
            self._dfs = [self._img_labels]
        elif isinstance(annotations_file, list):
            self._dfs = []
            for fn in annotations_file:
                self._dfs.append(pd.read_csv(fn))
            self._img_labels = pd.concat(self._dfs, ignore_index=True)

        self._mean_height = int(self._img_labels.iloc[:, 1].mean())
        self._mean_width = int(self._img_labels.iloc[:, 3].mean())

    @property
    def img_labels(self):
        """Returns combined annotations DataFrame."""
        return self._img_labels

    @property
    def mean_height(self):
        """Returns mean image height across dataset."""
        return self._mean_height

    @property
    def mean_width(self):
        """Returns mean image width across dataset."""
        return self._mean_width

    @property
    def dfs(self):
        """Returns list of annotation DataFrames if multiple loaded."""
        return self._dfs


class CrawlerImageDataset(Dataset):
    """
    PyTorch Dataset class that returns image tensors and labels.

    Args:
        img_labels (pd.DataFrame): DataFrame containing image file paths and labels.
        img_dir (str): Root folder where images are stored.
        transform (callable, optional): Transformations applied to images.
        target_transform (callable, optional): Transformations applied to labels.
    """
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns number of images in dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Returns an image tensor and its corresponding label.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: (image tensor, label tensor)
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")  # reliable image loading
        if self.transform:
            image = self.transform(image)
        label = self.img_labels.iloc[idx, 2]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MLModel:
    """
    Image classification model pipeline using PyTorch.

    Handles loading data, preparing datasets and dataloaders,
    defining, training, and evaluating an MLP neural network model.

    Args:
        annotations (str): CSV file name(s) with annotations inside dataset folder(s).
        data_folder (str): Path to root folder containing image datasets.
        query (str or list): Dataset folder name or list of folder names.
        validation_split (float): Fraction of dataset for validation (default: 0.3).
        shuffle_dataset (bool): Whether to shuffle data before splitting (default: True).
        learning_rate (float): Learning rate for optimizer (default: 1e-3).
        batch_size (int): Batch size for training and evaluation (default: 64).
        epochs (int): Number of training epochs (default: 20).
        model_out_fldr (str): Folder to save model (default: './model').
    """
    def __init__(self,
                 annotations: str,
                 data_folder: str,
                 query: str | list,
                 validation_split=0.3,
                 shuffle_dataset=True,
                 learning_rate=1e-3,
                 batch_size=64,
                 epochs=20,
                 model_out_fldr='./model'):
        # Determine dataset folder(s) and load annotations
        if isinstance(query, str):
            self.dataset_folder = os.path.join(data_folder, query)
            self.aloader = AnnotationsLoader(os.path.join(self.dataset_folder, annotations))
        elif isinstance(query, list):
            self.dataset_folder = [os.path.join(data_folder, q) for q in query]
            self.aloader = AnnotationsLoader([os.path.join(df, annotations) for df in self.dataset_folder])
        self.query = query
        # Determine mean image size for resizing
        self.img_size = (self.aloader.mean_height, self.aloader.mean_width)

        # Define image transformations: convert to PIL, resize, tensor
        transform = Compose([
            # ToPILImage(),
            Resize(self.img_size),
            ToTensor(),
        ])

        # Map labels to integer indices
        labs = sorted(set(self.aloader.img_labels.iloc[:, 2]))
        self.int_labels = {value: index for index, value in enumerate(labs)}

        # Target transform to convert labels to integer indices
        target_transform = Lambda(lambda y: torch.tensor(self.int_labels[y]))

        self.validation_split = validation_split
        self.shuffle_dataset = shuffle_dataset

        # Prepare training and validation datasets
        if isinstance(self.dataset_folder, str):
            train_indices, val_indices = self.split_indices(self.aloader.img_labels)
            self.train_data = CrawlerImageDataset(self.aloader.img_labels.iloc[train_indices], self.dataset_folder,
                                                  transform=transform, target_transform=target_transform)
            self.test_data = CrawlerImageDataset(self.aloader.img_labels.iloc[val_indices], self.dataset_folder,
                                                 transform=transform, target_transform=target_transform)
        elif isinstance(self.dataset_folder, list):
            train_datasets = []
            test_datasets = []
            for i in range(len(self.dataset_folder)):
                ind = self.aloader.dfs[i]
                df = self.dataset_folder[i]
                train_indices, val_indices = self.split_indices(ind)
                train_datasets.append(CrawlerImageDataset(ind.iloc[train_indices], df, transform=transform,
                                                          target_transform=target_transform))
                test_datasets.append(CrawlerImageDataset(ind.iloc[val_indices], df, transform=transform,
                                                         target_transform=target_transform))
            self.train_data = ConcatDataset(train_datasets)
            self.test_data = ConcatDataset(test_datasets)

        self.batch_size = batch_size
        # Create dataloaders for training and testing datasets
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle_dataset)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

        # Setup device: GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        # Compute input feature size for the MLP
        self.img_size = (self.aloader.mean_height, self.aloader.mean_width)
        in_features = 3 * self.img_size[0] * self.img_size[1]
        # Initialize neural network model
        self.model = NeuralNetwork(in_features, 512, len(self.int_labels)).to(self.device)
        print(self.model)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mod_out_fldr = model_out_fldr
        # CrossEntropyLoss expects logits and integer labels
        self.loss_fn = nn.CrossEntropyLoss()
        # Using Adam optimizer for training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def split_indices(self, img_labels):
        """
        Splits dataset indices into training and validation subsets.

        Args:
            img_labels (pd.DataFrame): DataFrame containing dataset annotations.

        Returns:
            tuple: (train_indices, validation_indices)
        """
        dataset_size = len(img_labels)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if self.shuffle_dataset:
            np.random.seed(int(time.time()))
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices

    def get_features_labels(self):
        train_features, train_labels = next(iter(self.train_dataloader))
        return train_features, train_labels

    def train_loop(self):
        """
            Executes one epoch of training over the training data.
            Prints loss metrics every 100 batches.
        """
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                current = batch * self.batch_size + len(X)
                print(f"Loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        """
        Evaluate the model on validation data.
        Prints accuracy and average loss.
        """
        self.model.eval()
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error:\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    def train_epochs(self):
        """
        Run training and evaluation loops for multiple epochs.
        """
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n--------------------------------------")
            self.train_loop()
            self.test_loop()
        print("Done!")

    def save_model(self, full_model: bool = False):
        """
        Save the model to disk at the path defined by `model_path()`.

        Args:
            full_model (bool): If True, saves the entire model object including architecture.
                If False (default), saves only the model's state dictionary (recommended).

        Notes:
            Saving only the state_dict is preferred for flexibility and compatibility.
            Full model saving uses Python pickle and can cause issues on refactor or different environments.
        """
        check_folder_mk(self.mod_out_fldr)
        if full_model:
            torch.save(self.model, self.model_path())
        else:
            torch.save(self.model.state_dict(), self.model_path())

    def load_model(self, model_path: str, weights_only: bool = True):
        """
        Load model weights from the specified path.

        Args:
            model_path (str): Path to the saved model file.
            weights_only (bool): If True, assumes file contains a state_dict and loads
                parameters into the existing model architecture.
                If False (default), assumes file contains entire model and tries to load it directly.

        Notes:
            For loading state_dict, the model must be initialized with the same architecture before calling this.
            Call `model.eval()` after loading for inference mode.
        """
        if weights_only:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            loaded_model = torch.load(model_path, weights_only=False)
            self.model = loaded_model

    def model_path(self) -> str:
        """
        Compute the file path for saving/loading the model.

        Combines the model output folder path and a hash based filename
        based on the current query and file extension 'pth'.

        Returns:
            str: Absolute path to model file.
        """
        return os.path.join(self.mod_out_fldr, hash_file_name('pth', self.query))


class NeuralNetwork(nn.Module):
    """
    Multi-layer perceptron for image classification.

    Args:
        in_features (int): Size of flattened input features.
        out_features (int): Number of units in hidden layers.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_features: int, out_features: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, num_classes)
        )

    def print_model_params(self):
        print(f"Model structure: {self.model}\n\n")
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")

    def forward(self, x):
        """
        Forward pass transforming input tensor to class logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (batch, num_classes).
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
