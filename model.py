!pip install rasterio

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms
import rasterio as rio
from rasterio.enums import Resampling


colors = np.array([0,1,2, 3])

def mapping(sample_image):
  #colors = np.unique(np.asarray(sample_image))
  target = torch.from_numpy(np.asarray(sample_image))
  target = target.permute(2, 0, 1).contiguous()

  mapping = {(0,): 0, (255,): 0, (175,): 2, (80,): 0}
  #print(mapping)
  mask = torch.empty(96,96, dtype=torch.long)
  for k in mapping:
      # Get all indices for current class
      idx = (target==torch.tensor(k, dtype=torch.float16).unsqueeze(1).unsqueeze(2))
      validx = (idx.sum(0) == 3)  # Check that all channels match
      mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
  return mask

import numpy as np
import math
import cv2
from skimage.color import rgb2lab, lab2rgb

class CleanWasteDataset(Dataset):

    def __init__(self, image_dir, mask_dir,class_rgb_values=None,
        augmentation=None,
        preprocessing=None):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.mask_dir = mask_dir
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = rio.open(image_fp).read(out_shape=(16,96,96),resampling=Resampling.bilinear)

        image = torch.Tensor(image)
        image = torch.nan_to_num(image)


        mask = Image.open(self.mask_dir+image_fn[:image_fn.index(".tif")]+".png").convert("RGB")
        mask = mask.resize((96,96),resample=Image.NEAREST)

        label_class = mapping(mask)
        '''for i in range(image.shape[0]):
          image[i,:,:][label_class==0]=0'''
        #print(image.shape)
        if self.augmentation:
            sample = self.augmentation(image=np.asarray(image.permute(1,2,0)), mask=np.asarray(label_class))
            #print(sample['image'].shape)

            image, label_class = sample['image'], sample['mask']
            image = np.asarray(torch.from_numpy(image).permute(2,0,1))

        #image = np.asarray(image.permute(2,0,1))
        return image, label_class

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform_ops(image)
mask_dir = "/content/drive/MyDrive/Data/RiverWasteMask/"
train_image_dir = "/content/drive/MyDrive/Data/SegmentationRivers/"
dataset = CleanWasteDataset(train_image_dir, mask_dir)
print(len(dataset))


from torch.utils.data import random_split

def get_images(batch_size=5,shuffle=False,pin_memory=True):
    test_split=0.2
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset =  random_split(dataset,
                                               [train_size, test_size])

    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return train_batch,test_batch

train_batch,test_batch = get_images(batch_size=5)
for img,mask in train_batch:

    #img1 = np.transpose(img[6,:,:,:],(1,2,0))
    mask1 = np.array(mask[0,:,:])
    #img2 = np.transpose(img[7,:,:,:],(1,2,0))
    mask2 = np.array(mask[1,:,:])
    #img3 = np.transpose(img[8,:,:,:],(1,2,0))
    mask3 = np.array(mask[2,:,:])
    fig , ax =  plt.subplots(3, 3, figsize=(18, 18))

    ax[0][0].imshow(3*(torch.stack((img[0,3],img[0,2],img[0,1])).permute(1,2,0).numpy().astype('float32')))
    ax[0][1].imshow(3*(torch.stack((img[0,10],img[0,8],img[0,7])) - torch.stack((img[0,9],img[0,9],img[0,9]))).permute(1,2,0).numpy().astype('float32'))
    ax[0][2].imshow(mask1)
    ax[1][0].imshow(3*torch.stack((img[1,3],img[1,2],img[1,1])).permute(1,2,0).numpy().astype('float32'))
    ax[1][1].imshow(3*(torch.stack((img[1,10],img[1,8],img[1,7])) - torch.stack((img[1,9],img[1,9],img[1,9]))).permute(1,2,0).numpy().astype('float32'))
    ax[1][2].imshow(mask2)
    ax[2][0].imshow(3*torch.stack((img[2,3],img[2,2],img[2,1])).permute(1,2,0).numpy().astype('float32'))
    ax[2][1].imshow(3*(torch.stack((img[2,10],img[2,8],img[2,7])) - torch.stack((img[2,9],img[2,9],img[2,9]))).permute(1,2,0).numpy().astype('float32'))
    ax[2][2].imshow(mask3)
    break

def labelToImg(out):
  out[out==1]=255
  out[out==2]=175
  out[out==3]=80
  return out

def show_rgb(msi):
  return 3*(torch.stack((msi[3],msi[2],msi[1])).permute(1,2,0).cpu().numpy().astype('float32'))



class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(int(embed_dim/2))
        self.attn = nn.MultiheadAttention(int(embed_dim/2), num_heads,
                                          dropout=dropout,bias=True)


        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        #print("In attn",x.shape)
        x = self.layer_norm_1(x.permute(0,2,3,4,1))

        B,H,W,D,C = x.shape
        x= x.view(B,H*W*D,C)
        x = x + self.attn(x, x, x)[0]

        return x.view(B,C,H,W,D)


class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []

        model.append(nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm3d(out_channels))
        model.append(nn.ReLU(inplace=True))
        '''model.append(nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm3d(out_channels))
        model.append(nn.ReLU(inplace=True))'''
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        #print(x.shape)
        return self.conv(x)

class decoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(decoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        '''model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))'''
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)

class unet_model(nn.Module):
    def __init__(self,out_channels=256,features=[16, 32, 64, 128]):
        super(unet_model,self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.pool_2d = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.pool_1 = nn.MaxPool3d(kernel_size=(3,3,1),stride=(3,3,1))
        self.conv1 = encoding_block(1,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])

        self.tconv2 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)

        self.t_conv2 = decoding_block(features[3], features[3])
        self.t_conv3 = decoding_block(features[2], features[2])
        self.t_conv4 = decoding_block(features[1], features[0])


        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)

        self.transformer0 = nn.Sequential(*[AttentionBlock(32, 64, 8, dropout=0.2) for _ in range(2)])
        self.transformer = nn.Sequential(*[AttentionBlock(64, 128, 8, dropout=0.2) for _ in range(2)])
        self.transformer1 = nn.Sequential(*[AttentionBlock(128, 256, 8, dropout=0.2) for _ in range(2)])
        self.transformer2 = nn.Sequential(*[AttentionBlock(256, 512, 8, dropout=0.2) for _ in range(2)])


    def forward(self,x):
        original = x
        skip_connections = []

        x = self.conv1(x)

        skip_connections.append(x.view(x.shape[0],x.shape[1]*x.shape[4],x.shape[2],x.shape[3]))
        #print(x.view(x.shape[0],x.shape[1],x.shape[2],-1).shape)


        x = self.pool(x)

        x = self.conv2(x)

        #x  = self.transformer0(x)

        skip_connections.append(x.view(x.shape[0],x.shape[1]*x.shape[4],x.shape[2],x.shape[3]))
        x = self.pool(x)

        x = self.conv3(x)

        #x  = self.transformer(x)

        skip_connections.append(x.view(x.shape[0],x.shape[1]*x.shape[4],x.shape[2],x.shape[3]))
        x = self.pool_2d(x)

        x = self.conv4(x)

        #x  = self.transformer1(x)
        #print(x.shape)
        skip_connections.append(x.view(x.shape[0],x.shape[1]*x.shape[4],x.shape[2],x.shape[3]))

        x  = self.transformer2(x)
        x = self.bottleneck(x)
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][0].detach().cpu())'''

        #print(x.shape)
        x = x.view(x.shape[0],x.shape[1]*x.shape[4],x.shape[2],x.shape[3])
        skip_connections = skip_connections[::-1]
        #print(x.shape)
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][0].detach().cpu())'''

        #print(x.shape,skip_connections[0].shape)

        #x = torch.cat((skip_connections[0], x), dim=1)
        #print(x.shape)
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][0].detach().cpu())'''
        #print(x.shape)
        x = self.tconv2(x)
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][0].detach().cpu())'''
        #print(x.shape, skip_connections[1].shape)
        #x = torch.cat((skip_connections[0], x), dim=1)

        #print(x.shape)
        x = self.t_conv2(x)

        #print(x.shape)

        x = self.tconv3(x)
        #print(x.shape)
        #x = torch.cat((skip_connections[1], x), dim=1)


        x = self.t_conv3(x)
        x = self.tconv4(x)
        #x = torch.cat((skip_connections[2], x), dim=1)
        x = self.t_conv4(x)

        x = self.final_layer(x)
        #print(x[0].shape)
        '''castConvOutputs = x.type(torch.FloatTensor)
        cam = torch.mean(castConvOutputs, axis=(0))

        heatmap1 = cv2.resize(cam.detach().cpu().numpy(), (cam.shape[1], cam.shape[2]))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap1 - np.min(heatmap1)
        denom = (heatmap1.max() - heatmap1.min()) + 0.000000000001
        heatmap1 = numer / denom
        '''
        #print(x.shape)
        return x


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
modele = unet_model(out_channels=2).to(DEVICE)
from torchsummary import summary
summary(modele, (1, 96, 96,12))
save_path = '/content/drive/MyDrive/unet_attn.pth'




#model = UNet(num_classes=4).to(DEVICE)
min_loss = 1000
#200 epochs

def multi_acc(pred, label):
    _, tags = torch.max(pred, dim = 1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


import torch.optim as optim
batch_size = 1

epochs = 20
lr = 3e-4

modele = unet_model(out_channels=4).to(DEVICE)
model = modele
#criterion = DiceLoss(weight=[0,1,3,2])
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.001,0,0.6,0]).to(DEVICE))
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()

step_losses = []
epoch_losses = []
val_acc = 0

for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    accuracy = 0
    for X, Y in tqdm(train_batch, total=len(train_batch), leave=False):
        X, Y = X.to(DEVICE), Y.to(DEVICE)


        Y_pred = model(X[:,0:12].unsqueeze(1).permute(0,1,3,4,2))
        #loss = criterion(Y_pred.squeeze(1).to(DEVICE), Y.type(torch.FloatTensor).to(DEVICE) )

        loss = criterion(Y_pred.squeeze(1).to(DEVICE), Y.to(DEVICE) )
        accuracy += multi_acc(Y_pred.squeeze(1).to(DEVICE), Y.to(DEVICE))
        #print(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        step_losses.append(loss.item())
    epoch_losses.append(epoch_loss/len(train_batch))
    if epoch_loss/len(train_batch) < min_loss:
      min_loss = epoch_loss/len(train_batch)
      best_dict = model.state_dict()

    print("Epoch ",epoch," Avg loss ",epoch_loss/len(train_batch),"Train Accuracy", accuracy/len(train_batch))

    test_accuracy = 0
    for X_test, Y_test in tqdm(test_batch, total=len(test_batch), leave=False):
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)


        Y_pred_test = model(X_test[:,0:12].unsqueeze(1).permute(0,1,3,4,2))
        #loss = criterion(Y_pred.squeeze(1).to(DEVICE), Y.type(torch.FloatTensor).to(DEVICE) )
        test_accuracy += multi_acc(Y_pred_test.squeeze(1).to(DEVICE), Y_test.to(DEVICE))
    print("Test Accuracy", test_accuracy/len(test_batch))
    if test_accuracy>val_acc:
      best_val_dict = model.state_dict()


