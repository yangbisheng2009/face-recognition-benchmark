import yaml
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T
import argparse

from utils.models import *
from utils.utils import extract_feature, cosin_metric

parser = argparse.ArgumentParser(description='Create FaceDB')
parser.add_argument('-p', '--project', default='./configs/ccfb.yaml', help='project file')
parser.add_argument('--checkpoint', default='./checkpoints/92-98.9566438832252.pth', help='checkpoint')
parser.add_argument('--db', default='./db/train-driver.pth', help='checkpoint')
parser.add_argument('--input-images', default='./input-images', help='input images')
args = parser.parse_args()

def main():
    model = torch.load(args.checkpoint)['model']
    torch.save(model.module.state_dict(), './fc.pth')

if __name__ == '__main__':
    main()
