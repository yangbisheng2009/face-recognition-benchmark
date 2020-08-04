import os
import sys
import cv2
import yaml
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
from utils.utils import extract_feature
from torchvision import transforms as T


parser = argparse.ArgumentParser(description='Create FaceDB')
parser.add_argument('-p', '--project', default='./configs/ccfb.yaml', help='project file')
parser.add_argument('--checkpoint', default='./checkpoints/ccfb/78-98.72374111831944.pth', help='checkpoint')
args = parser.parse_args()

def main():
    with open(args.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    project_name = data_dict['project_name']
    is_aver = data_dict['is_aver']
    faces_dir = data_dict['faces_dir']
    db_dir = data_dict['db_dir']
    os.makedirs(db_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = torch.load(args.checkpoint)['model'].module.to(device)
    else:
        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model'].module.to(device)
    model.eval()

    tsfm = T.Compose([T.Resize((112,112)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    db_dict = defaultdict()
    for folder in tqdm(os.listdir(faces_dir)):
        if is_aver:
            pass
        else:
            faces = os.listdir(os.path.join(faces_dir, folder))
            assert(len(faces) > 0)
            feature = extract_feature(model, tsfm, os.path.join(faces_dir, folder, faces[0]), device)[0]
            db_dict[folder] = feature

    print('The lenth of Face DB: {}'.format(len(db_dict)))
    torch.save(db_dict, os.path.join(db_dir, project_name+'.db'))


if __name__ == '__main__':
    main()
