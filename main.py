import torch 
import argparse 
import numpy as np 
from utils import *
from model import *
from tqdm import tqdm 
from dataset import dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description = "Do Stuff")

parser.add_argument("--image_size", type = int, default = 352)
parser.add_argument("--threshold", type = float, default = 1e-5)
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--weight_decay", type = float, default = 1e-4)
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument("--epoch", type = int, default = 200)
parser.add_argument("--train_dir", type = str, default = 'polyp/TrainDataset')
parser.add_argument("--test_dir", type = str, default = 'polyp/TestDataset')
parser.add_argument("--test_dataset", type = str, default = 'Kvasir')
parser.add_argument("--SEED", type = int, default = 28)

args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)

    train_data = dataset(args, mode = 'train')
    test_data = dataset(args, mode = 'test')
    train_loader = DataLoader(dataset=train_data, batch_size = args.batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size = 1)
    
    model = Model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay= args.weight_decay)
    
    train(model, train_loader, test_loader, bce_dice_loss, optimizer, args.epoch)
