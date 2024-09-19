import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob

SEED = 28
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

transform = transforms.Compose([
                  transforms.Resize((352, 352)),
                  transforms.ToTensor(),
                                ])

class dataset(Dataset):
  def __init__(self, image_path, label_path, transform):
    self.image_path = sorted(glob.glob(image_path+'/*'))
    self.label_path = sorted(glob.glob(label_path+'/*'))
    self.transform = transform
    
  def __len__(self):
    return len(self.label_path)

  def __getitem__(self,index):
    image = Image.open(self.image_path[index])
    label = Image.open(self.label_path[index])
    label = label.convert('L')
    image = self.transform(image)
    label  = self.transform(label )
    label[label >=0.5] = 1
    label[label <0.5] = 0
    return image, label 

train_dataset = dataset(image_path = 'polyp/TrainDataset/images' , label_path = 'polyp/TrainDataset/masks', transform = transform)
validation_dataset = dataset(image_path = 'polyp/TestDataset/Kvasir/images' , label_path = 'polyp/TestDataset/Kvasir/masks', transform = transform)
test_dataset = dataset(image_path = 'polyp/TestDataset/Kvasir/images'  , label_path = 'polyp/TestDataset/Kvasir/masks', transform = transform)

train_loader = DataLoader(dataset = train_dataset , batch_size = 16)
validation_loader = DataLoader(dataset = validation_dataset , batch_size = 1)
test_loader = DataLoader(dataset = test_dataset , batch_size = 1)

