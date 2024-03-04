import numpy as np
from sklearn.utils import class_weight

# Load the label files
train_labels = np.loadtxt('/home/mwilkers1/Documents/Projects/IMPACT-Edge-AI/data/labels/train-set-v2.1.txt', dtype=str)
val_labels = np.loadtxt('/home/mwilkers1/Documents/Projects/IMPACT-Edge-AI/data/labels/val-set-v2.1.txt', dtype=str)
test_labels = np.loadtxt('/home/mwilkers1/Documents/Projects/IMPACT-Edge-AI/data/labels/test-set-v2.1.txt', dtype=str)

# Extract class labels
train_labels = train_labels[:, -1].astype(int)
val_labels = val_labels[:, -1].astype(int)
test_labels = test_labels[:, -1].astype(int)

# Combine all labels
all_labels = np.concatenate((train_labels, val_labels, test_labels))

# Compute class distribution
class_distribution = np.bincount(all_labels)

# Calculate the weight for each class, avoiding division by zero
class_weights = np.zeros_like(class_distribution, dtype=np.cfloat)
non_zero_indices = class_distribution != 0
class_weights[non_zero_indices] = len(all_labels) / (len(np.unique(all_labels)) * class_distribution[non_zero_indices])
# Extract the real parts of the class weights
class_weights_real = np.real(class_weights)

print("Class Weights:")
print(class_weights_real)