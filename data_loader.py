import numpy as np
import pickle
import os
import sys
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
### Data loading
# Function to load a single batch file
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
        return X, Y


# Function to load the test batch without labels
def load_cifar_test_nolabels(filename):
    with open(filename, 'rb') as f:
        try:
            datadict = pickle.load(f, encoding='bytes')
            # print("Data dictionary keys:", datadict.keys())  # To check the structure of the dictionary
            # Access the correct key for the data
            X = datadict[b'data']
            X = X.reshape(-1, 3, 32, 32).astype('float32')  # Adjust shape accordingly if needed
            return X
        except EOFError as e:
            print("EOFError while unpickling the file:", e)
        except Exception as e:
            print("An error occurred:", e)

# Function to load the test batch
def load_cifar_test_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict.get(b'labels', None)  # Try to get the labels if they exist
        X = X.reshape(-1, 3, 32, 32).astype("float32")  # Reshape and convert data type
        return X, Y

class CIFAR10Dataset(Dataset):
    def __init__(self, data, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns one sample at a time
        """
        if index >= len(self.data):
            raise IndexError(f'Requested index {index} exceeds dataset length {len(self.data)}.')
        # Select sample
        image = self.data[index]

        # Convert image from numpy array to PIL Image to apply transform
        image = Image.fromarray(image.astype('uint8').transpose((1, 2, 0)))

        # Apply the given transform
        if self.transform:
            image = self.transform(image)

        # Return image and the corresponding label if available
        if self.targets is not None:
            target = self.targets[index]
            return image, target
        else:
            return image
