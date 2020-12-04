import yaml
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T
import argparse

from utils.dataset import Dataset
from utils.utils import AverageMeter, accuracy, calc_num_person
from models.focal_loss import FocalLoss
from models.optimizer import InsightFaceOptimizer
from models.models import resnet18, resnet34, resnet50, resnet101, resnet152, MobileNet, resnet_face18, ArcMarginModel

parser = argparse.ArgumentParser(description='Strip checkpoint to make it smaller.')
parser.add_argument('--input-checkpoint', default='', help='input checkpoint')
parser.add_argument('--output-model', default='', help='output model')
parser.add_argument('--output-metric_fc', default='', help='output metric_fc')
args = parser.parse_args()

def main():
    model = torch.load(args.input_checkpoint)['model']
    metric_fc = torch.load(args.input_checkpoint)['metric_fc']
    torch.save(model.module.state_dict(), args.output_checkpoint)
    torch.save(metric_fc.module.state_dict(), args.output_metric_fc)
    

if __name__ == '__main__':
    main()
