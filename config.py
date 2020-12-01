import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of

# Data parameters
num_classes = 1728
num_samples = 3804846
DATA_DIR = 'data'
faces_ms1m_folder = 'data/faces_ms1m_112x112'
path_imgidx = os.path.join(faces_ms1m_folder, 'train.idx')
path_imgrec = os.path.join(faces_ms1m_folder, 'train.rec')
IMG_DIR = 'data/images'
pickle_file = 'data/faces_ms1m_112x112.pickle'
