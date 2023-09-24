# --------------------------------------------------------
# train code modefied from yunjey/pytorch-tutorial
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
# By yhx
# --------------------------------------------------------

import torch 
import torch.nn as nn
import torchvision
# from pandas.util._exceptions import find_stack_level
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from rgbd_vla import RGBD2pose
from torch.utils.data import DataLoader
from custom_6dpose_dataset import IN2POSEDATASET
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# def custom_collate_fn(batch):
#     images = [torch.from_numpy(image) for image in batch]  # 将PIL.Image.Image转换为张量
#     return torch.stack(images)
    # return collate([torch.from_numpy(b) for b in batch])
def custom_collate_fn(batch):
    # 将numpy.uint16类型的数据转换为支持的数据类型
    batch[2] = [torch.tensor(data[2].astype(np.float32)) for data in batch]
    return batch

# Hyper parameters
num_epochs = 500
batch_size = 8 # 100
learning_rate = 0.001

# data = pd.read_csv('/data/ML_document/datasets/custom_6dpose_dataset/custom_6dpose_dataset.csv')

# # 分割数据集
# train_dataset = data[data['split'] == 'train']
# val_dataset = data[data['split'] == 'val']

data = IN2POSEDATASET()
train_dataset, val_dataset = torch.utils.data.random_split(data, [48, 13])
# train_dataset, val_dataset = torch.utils.data.random_split(data, [1, 60])

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True, 
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)
                                        #    collate_fn=custom_collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
                                        #   collate_fn=custom_collate_fn)

model = RGBD2pose(device=device, lr=1e-3, hidden_dim=520)

# Loss and optimizer
update_params = []
for param in model.e2pose.parameters():
    update_params.append(param)
update_params += list(model.dep_fea_extra.depth_feature_extract.fc.parameters())
# update_params = [model.e2pose, model.dep_fea_extra.depth_feature_extract.fc]

criterion = nn.CrossEntropyLoss()
criterion_1 = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(update_params, lr=learning_rate)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, data_detailed in enumerate(train_loader):
        # (instructions, images_rgb, images_depth, outputs)
        instructions, images_rgb, images_depth, outputs = data_detailed
        instructions = list(instructions)
        images_rgb = images_rgb.to(device)
        images_depth = images_depth.to(device)
        labels = outputs.to(device)
        # labels = torch.tensor(labels).long()
        
        # Forward pass
        outputs = model(images_rgb, images_depth, instructions, device=device)
        loss = criterion_1(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 6 == 0:
            with open('./loss_log.txt', 'a') as f:
                f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        with open('./loss_log.txt', 'a') as f:
            f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    # Test the model
    if epoch % 10 == 0:
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            error = 0
            for instructions, images_rgb, images_depth, outputs in test_loader:
                images_rgb = images_rgb.to(device)
                images_depth = images_depth.to(device)
                labels = outputs.to(device)
                
                outputs = model(images_rgb, images_depth, instructions, device=device)
                loss = criterion_1(outputs, labels)
                
                total += labels.size(0)
                error += loss
            with open('./loss_log.txt', 'a') as f:
                f.write('The average error of the model on the 13 test images  is: {} '.format(error / total))

            print('The average error of the model on the 13 test images  is: {} '.format(error / total))
f.close()

# Save the model checkpoint
torch.save(model.state_dict(), './model.ckpt')