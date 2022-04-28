import glob
from sklearn.ensemble import VotingRegressor
import torch
from torch._C import device 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
from skimage import io, transform
from torchvision.transforms import transforms
from torchvision import utils
import numpy as np
from torchvision import datasets,models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from efficientnet_pytorch import EfficientNet
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score 
from tqdm import tqdm 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# totalLables = np.array(os.listdir('./torchData/train'))

with open('./labels.txt', 'r', encoding='utf-8') as f:
  totalLables = []
  for i in f:
    totalLables.append(i.strip())

nLabels = len(totalLables)

  
class gemStone(Dataset):
    def __init__(self, csv_file, transform):
        self.train_fram = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.train_fram)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(os.path.join(self.root, self.train_fram.values[idx][0]))
        image = cv2.imread(self.train_fram.values[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        label = self.train_fram.values[idx, 1]
        nplabel = np.where(totalLables==label, 1.0, 0.0)
        labels = np.argmax(nplabel)
        # nplabel = nplabel.astype(float)
        # nplabel = nplabel.astype(np.double)
        # print(label)
        # label = label.astype(float)

        if self.transform :
            image = self.transform(image=image)['image']
        return image, labels

if __name__ == '__main__':
  transform_train = A.Compose([
    # A.Resize(224, 224), 
    A.RandomBrightnessContrast(brightness_limit=0.1, 
                               contrast_limit=0.1, p=0.5),
    A.VerticalFlip(p=0.5), 
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=25, p=0.7),

    A.OneOf([A.Emboss(p=1),
             A.Sharpen(p=1),
             A.Blur(p=1)], p=0.5),
    A.PiecewiseAffine(p=0.5), 
    A.Normalize(), 
    ToTensorV2()
  ])

  transform_val = A.Compose([
      # A.Resize(224, 224), 
      A.Normalize(), 
      ToTensorV2()
  ])


  gemStoneDataset_train = gemStone('train.csv', transform_train)
  gemStoneDataset_val = gemStone('val.csv', transform_val)

  g = torch.Generator()
  g.manual_seed(0)

  train_loader = DataLoader(gemStoneDataset_train, batch_size=128, shuffle=True, num_workers=12, generator=g)
  val_loader = DataLoader(gemStoneDataset_val, batch_size=128, shuffle=False, num_workers=12, generator=g)

  model = models.vgg16(pretrained=True)

  model.classifier[6] = nn.Linear(4096, 513)
  print(model)
  # model = nn.DataParallel(model)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.000007, weight_decay=0.00001)

  # model.load_state_dict(torch.load('stone_b7_full.pth'), strict=False)
  model = model.to(device=device)



  epochs = 200


  scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=len(train_loader)*5, 
                                              num_training_steps=len(train_loader)*epochs)


  max_acc_5 = 0
  for epoch in range(epochs):
      model.train() 
      epoch_train_loss = 0 

      for image, label in tqdm(train_loader):

          image = image.to(device)
          label = label.to(device)
          optimizer.zero_grad()

          outputs = model(image)

          loss = criterion(outputs, label)
          loss.backward()
          optimizer.step()
          scheduler.step() 
          epoch_train_loss += loss.item() 

      model.eval()
      epoch_val_loss = 0
      correct_5 = 0
      correct_1 = 0
      map5_metrics = 0
      with torch.no_grad(): 
          for image, label in tqdm(val_loader):
              image = image.to(device)
              label = label.to(device)
              outputs = model(image)
              
              pred_1 = outputs.argmax(dim=1)
              correct_1 += torch.eq(pred_1, label).sum().float().item()
              maxk = max((1,5))
              _, pred = outputs.topk(maxk, 1, True, True)
              label_resize = label.view(-1, 1)
              loss = criterion(outputs, label)
              correct_5 += torch.eq(pred, label_resize).sum().float().item()
              epoch_val_loss += loss.item()
              mapColo = torch.eq(pred, label_resize)
                      # print(mapColo)
              npMapColo = mapColo.nonzero().cpu().detach().numpy()
              for tupple in npMapColo:
                  map5_metrics += 1/(tupple[1]+1.0)

      print(f'Epoch [{epoch+1}/{epochs}] - Train data loss : {epoch_train_loss/len(train_loader):.4f}')
      print(f'Epoch [{epoch+1}/{epochs}] - Val data loss : {epoch_val_loss/len(val_loader):.4f}')
      print(f'Epoch [{epoch+1}/{epochs}] - ACC_1 : {correct_1/len(gemStoneDataset_val)*100}%')
      print(f'Epoch [{epoch+1}/{epochs}] - ACC_5 : {correct_5/len(gemStoneDataset_val)*100}%')
      print(len(train_loader), " ", len(val_loader), " ", len(gemStoneDataset_val))
      if not os.path.exists(f'checkpoints/vgg/{epoch + 1}'):
          os.mkdir(f'checkpoints/vgg/{epoch + 1}')
      with open(f'checkpoints/vgg/{epoch + 1}/metrics.txt', "w") as f:
              f.write(f'Epoch [{epoch+1}/{epochs}] - Train data loss : {epoch_train_loss/len(train_loader):.4f}\n')
              f.write(f'Epoch [{epoch+1}/{epochs}] - Val data loss : {epoch_val_loss/len(val_loader):.4f}\n')
              f.write(f'Epoch [{epoch+1}/{epochs}] - ACC_1 : {correct_1/len(gemStoneDataset_val)*100}%\n')
              f.write(f'Epoch [{epoch+1}/{epochs}] - ACC_5 : {correct_5/len(gemStoneDataset_val)*100}%\n')
              f.write(f'Epoch [{epoch+1}/{epochs}] - MAP@5 : {map5_metrics}\n')
      if (correct_5 > max_acc_5) :
          torch.save(model.module.state_dict(), './mushroom_best_model.pth')
          max_acc_5 = correct_5
  torch.save(model.module.state_dict(), './mushroom_last_model.pth')


