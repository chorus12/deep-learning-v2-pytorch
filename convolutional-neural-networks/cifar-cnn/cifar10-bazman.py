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

# !jupytext --to py cifar10-bazman.ipynb

# # CIFAR 10 network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import RandomSampler, SubsetRandomSampler
from torch.utils.data import DataLoader
from os.path import join, abspath, dirname
from tqdm.notebook import tqdm
from torchvision.utils import make_grid

transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
data = datasets.CIFAR10(root='/home/testuser/backup_folder/data/cifar10', download=True, transform=transform)

import random
import numpy as np
N = len(data); print(N)
valid_split = 0.2
N_VALID = int(N*valid_split); print(N_VALID)
indices = list(range(N))
random.shuffle(indices)
train_shuffle, valid_shuffle = SubsetRandomSampler(indices[N_VALID:]), SubsetRandomSampler(indices[:N_VALID])

len(train_data_loader.sampler)

batch_size = 500
train_data_loader = DataLoader(dataset=data, batch_size=batch_size, sampler=train_shuffle)
valid_data_loader = DataLoader(dataset=data, batch_size=batch_size, sampler=valid_shuffle)

batch = iter(train_data_loader)
imgs, labels = next(batch);

import PIL
def plot_grid_from_tensors(imgs):
    unnormalize = T.Normalize((1, 1, 1),(2,2,2), inplace=True)
    pils = imgs.clone().detach()
    for pil in pils:
        unnormalize(pil)
    return T.ToPILImage()(make_grid(pils))     


grid = plot_grid_from_tensors(imgs[:8])
print(*[data.classes[i] for i in labels.numpy()[:8]], sep='       ')
grid.resize(np.array(grid.size)*3, PIL.Image.BILINEAR)

imgs.shape

device = 'cuda' if torch.cuda.is_available() else 'cpu';print (device)


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=82, kernel_size=(3,3), padding=1, stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.dropout = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(738, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(self.dropout(x)))
        x = self.pooling(x)
        x = F.relu(self.conv2(self.dropout(x)))
        x = self.pooling(x)
        x = F.relu(self.conv3(self.dropout(x)))
        x = self.pooling(x)
        batch, dim1, dim2 = x.shape[-3:]
        x = x.view(-1, batch*dim1*dim2)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x


# +
model = CifarNet()
min_loss = np.inf

out = model(imgs[:8])
out.shape
# -

torch.backends.cudnn.enabled

# +
N_EPOCH = 30

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters())

model.to(device)

for epoch in tqdm(range(N_EPOCH)):
    
    running_loss = 0 
    model.train()
    # train one epoch
    for imgs, labels in train_data_loader:     
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        scores = model(imgs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    running_loss = running_loss/len(train_data_loader.sampler)
    
    # get the loss on the validation set
    model.eval()
    eval_loss = 0 
    with torch.no_grad():
        for imgs, labels in valid_data_loader:     
            imgs, labels = imgs.to(device), labels.to(device)
            scores = model(imgs)
            loss = criterion(scores, labels)
            eval_loss += loss.item()
        
    eval_loss = eval_loss/len(valid_data_loader.sampler)
    
    # make a print out
    print(f"Epoch {epoch:2}\t Train loss {running_loss:.4f}\tValid loss {eval_loss:.4f}")
    if eval_loss < min_loss:
        print("Nice :) Loss decreasing... Saving model")
        torch.save(model.state_dict(), 'cifar.pt')
        min_loss = eval_loss
# -

model_params = torch.load('cifar.pt')
model.load_state_dict(model_params)

# ## Calc the metrics

batch_size = 1000
test_data = datasets.CIFAR10(root='/home/testuser/backup_folder/data/cifar10', download=True, transform=transform, train=False)
# train_data_loader = DataLoader(test_data, batch_size=len(test_data))
test_data_loader = DataLoader(test_data, batch_size=batch_size)

# +
model.eval()
positives = 0
positives_by_class = [0. for item in test_data.classes]
class_total = [0 for item in test_data.classes]
with torch.no_grad():
    for imgs, labels in test_data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        scores = model(imgs)
        _, pred_labels = torch.max(scores, dim=1)
        positives += (pred_labels == labels).to('cpu').numpy().sum()
        
        for cifar_class_id in range(len(test_data.classes)):
            pred_class_idx = set((pred_labels == cifar_class_id).nonzero().flatten().cpu().numpy())
            target_class_idx = set((labels == cifar_class_id).nonzero().flatten().cpu().numpy())
            positives_by_class[cifar_class_id] += len(target_class_idx.intersection(pred_class_idx))
            class_total[cifar_class_id] += len(target_class_idx)
        
for i, (class_positives, cifar_class) in enumerate(zip(positives_by_class, test_data.classes)):
    print(f' {cifar_class:15} \t {class_positives/class_total[i]:.0%} = {class_positives:.0f}/{class_total[i]}')
print(f'\nOverall accuracy is {positives/len(test_data):.1%} = {sum(positives_by_class):.0f}/{len(test_data)}')
