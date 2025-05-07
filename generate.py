import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import random
from PIL import Image

# Download CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
x_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True).data

minority_class = 3  # 'cat' class

minority_data = [(image, label) for image, label in trainset if label == minority_class]
majority_data = [(image, label) for image, label in trainset if label != minority_class]

# Let's imbalance the 'cat' class by taking only 500 instances
minority_data = minority_data[:500]

imbalanced_data = minority_data + majority_data

# Fix seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Patch parameters: x% area, y% of the dataset
patch_area_frac = 0.1  # portion of image area to cover
apply_frac = 0.01       # portion of dataset to patch
num_to_patch = int(len(imbalanced_data) * apply_frac)
indices = random.sample(range(len(imbalanced_data)), num_to_patch)

for idx in indices:
    image, label = imbalanced_data[idx]
    arr = np.array(image)
    h, w, _ = arr.shape
    patch_area = int(patch_area_frac * h * w)
    patch_size = int(np.sqrt(patch_area))
    x1 = random.randint(0, w - patch_size)
    y1 = random.randint(0, h - patch_size)
    arr[y1:y1+patch_size, x1:x1+patch_size] = 0
    imbalanced_data[idx] = (Image.fromarray(arr), label)

print(f"{num_to_patch} images were patched.")

# Save imbalanced dataset
with open('imbalanced_data.pkl', 'wb') as f:
    pickle.dump(imbalanced_data, f)

with open('test_data.pkl', 'wb') as f:
    pickle.dump(x_test, f)

print(f"Imbalanced dataset created with {len(imbalanced_data)} samples. Data saved to 'imbalanced_data.pkl'")
print(f"Test dataset created with {len(x_test)} samples. Data saved to 'test_data.pkl'")
