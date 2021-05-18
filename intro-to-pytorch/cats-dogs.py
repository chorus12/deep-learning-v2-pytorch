# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Cats vs Dogs

# !jupytext --to py cats-dogs.ipynb

# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
import os

ROOT = os.path.abspath(os.path.curdir)
DATA_ROOT = os.path.join(ROOT, 'data')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
TEST_ROOT = os.path.join(DATA_ROOT, 'test')


class ImageFolderPath(datasets.ImageFolder):
    """
    class that returns the image filename as well
    """
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        filename = self.imgs[index]
        return (original_tuple[0], original_tuple[1], filename)


# +
train_transform = T.Compose([T.RandomRotation(30), 
                             T.RandomResizedCrop(224), 
                             T.RandomHorizontalFlip(), 
                             T.ToTensor(), 
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

test_transform = T.Compose([T.Resize(244), 
                            T.CenterCrop(224), 
                            T.ToTensor(), 
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

unnormalize = T.Compose([T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])])

train_dataset = ImageFolderPath(TRAIN_ROOT, transform=train_transform)

test_dataset = ImageFolderPath(TEST_ROOT, transform=test_transform)
# -

train_dataset.classes, train_dataset

train_data = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)

imgs, labels, paths = next(iter(train_data))


def unnorm_abunch(tensor_array):
    '''
    gets an array of tensors and applies unnormalize 
    '''
    from copy import deepcopy
    im_grid = deepcopy(tensor_array)
    for image_idx in range(tensor_array.shape[0]):
        im_grid[image_idx] = unnormalize(im_grid[image_idx])
    return im_grid


T.ToPILImage()(make_grid(unnorm_abunch(imgs[:8])))

labels[:8]

test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)

test_imgs, test_labels, test_paths = next(iter(test_data))

T.ToPILImage()(make_grid(unnorm_abunch(test_imgs[:8])))

test_labels[:8]

# ## Let's make some transfer learning

from torchvision.models import resnet50

model = resnet50(pretrained=True)


def number_of_parameters(net):
    """
    get the number of parameters in a network
    """
    return sum((list(map(lambda x: torch.as_tensor(x.flatten().size()).sum().item(), net.parameters())))) 


print(f"Number of parameters in a network {number_of_parameters(model):,}")

# + jupyter={"outputs_hidden": true}
for item in model.named_parameters():
    print(item[0])
# -

# make all parameters fixed
model.requires_grad_(False);

# + jupyter={"outputs_hidden": true}
for item in model.named_parameters():
    print(item[0], item[1].requires_grad)
# -

# make a new classifier
from collections import OrderedDict
model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features=2048, out_features=256)),
    ('fc1_activation', nn.ReLU()),
    ('fc1_dropout', nn.Dropout(p=0.4)),
    ('fc2', nn.Linear(in_features=256, out_features=2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.fc.requires_grad_(True)

# ## Make a training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device);

# !nvidia-smi

# +
N_EPOCH = 3

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters())
mini_batch_cnt = 0

# give output every number of minibatches
report_every = 10

running_loss = 0

for epoch in tqdm(range(N_EPOCH)):

    model.train()
    for images, labels, path in train_data:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predict = model.forward(images)
        loss = criterion(predict, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        mini_batch_cnt +=1
        
        # make an output of metrics
        if mini_batch_cnt % report_every == 0:
            model.eval()
            with torch.no_grad():
                accuracy = 0.0
                test_loss = 0.0
                
                for images, labels, path in train_data:
                    images, labels = images.to(device), labels.to(device)
                    predict_log_probs = model(images)
                    batch_loss = criterion(predict_log_probs, labels)
                    test_loss += batch_loss.item()
                    _, predict_class = predict_log_probs.max(dim=1)
                    equality = predict_class == labels
                    accuracy += equality.sum().to('cpu').item()
            
            accuracy /= len(train_dataset)
            print(f'''Epoch: {epoch}
            Mini-batch: {mini_batch_cnt}
            Running loss: {running_loss/report_every:.3f}
            Test loss: {test_loss/len(train_data):.3f}
            Accuracy on test : {accuracy:.2%}
            ----------------------------------''')
            running_loss = 0
            model.train()
