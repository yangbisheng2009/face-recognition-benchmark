import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import time
import yaml
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T

from utils.models import *
from utils.utils import extract_feature, cosin_metric

parser = argparse.ArgumentParser(description='Create FaceDB')
parser.add_argument('-p', '--project', default='./configs/ccfb.yaml', help='project file')
parser.add_argument('--checkpoint', default='./checkpoints/92-98.9566438832252.pth', help='checkpoint')
parser.add_argument('--db', default='./db/train-driver.pth', help='checkpoint')
parser.add_argument('--input-images', default='./input-images', help='input images')
args = parser.parse_args()


def main():
    with open(args.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    project_name = data_dict['project_name']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = torch.load(args.checkpoint)['model'].module.to(device)
    else:
        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model'].module.to(device)
    model.eval()

    tsfm = T.Compose([T.Resize((112,112)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load face DB
    face_db = torch.load(args.db)

    for fname in os.listdir(args.input_images):
        img_path = os.path.join(args.input_images, fname)
        feature = extract_feature(model, tsfm, img_path, device)[0]

        max_score = 0
        max_label = ''
        for k, v in face_db.items():
            score = cosin_metric(feature, v)
            label = k

            if score > max_score and score > 0.55:
                max_score = score
                max_label = k
        print('Recognize fname is: {}, its score is: {}'.format(max_label, max_score))


if __name__ == '__main__':
    main()
