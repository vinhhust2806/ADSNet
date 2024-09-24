from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob


class dataset(Dataset):
  def __init__(self, args, mode = 'train'):
    if mode == 'train':
      self.image_path = sorted(glob.glob(args.train_dir +'/images/*'))
      self.mask_path = sorted(glob.glob(args.train_dir +'/masks/*'))

    elif mode == 'test':
      self.image_path = sorted(glob.glob(args.test_dir + '/' + args.test_dataset + '/images/*'))
      self.mask_path = sorted(glob.glob(args.test_dir + '/' + args.test_dataset + '/masks/*'))
    
    #print(self.mask_path)
    self.transform = transforms.Compose([
                  transforms.Resize((args.image_size, args.image_size)),
                  transforms.ToTensor()])
    
  def __len__(self):
    return len(self.mask_path)

  def __getitem__(self,index):
    image = Image.open(self.image_path[index])
    mask = Image.open(self.mask_path[index])
    mask = mask.convert('L')
    image = self.transform(image)
    mask = self.transform(mask)
    mask[mask >=0.5] = 1
    mask[mask <0.5] = 0
    return image, mask 


