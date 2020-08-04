import os
import yaml
import argparse
from collections import defaultdict
from tqdm import tqdm
import random
import torch
from torchvision import transforms as T

from utils.utils import extract_feature, cosin_metric

parser = argparse.ArgumentParser(description='Create FaceDB')
parser.add_argument('-p', '--project', default='./configs/true-name.yaml', help='project file')
parser.add_argument('--which', default='db', help='which to val')
parser.add_argument('--checkpoint', default='./checkpoints/true-name/283-99.39035238751148-r101.pth', help='checkpoint')
parser.add_argument('--db', default='./db/true-name.db', help='face db')
args = parser.parse_args()

def val_db(faces_dir, model, tsfm, device):
    print('Load face DB...')

    RIGHT = 0.0
    WRONG = 0.0
    UNKNOW = 0.0
    CNT = 0.0
    
    small_p = []

    db = torch.load(args.db)
    for folder in tqdm(os.listdir(faces_dir)):
        true_label = folder
        for f in os.listdir(os.path.join(faces_dir, folder)):
            CNT += 1
            feature = extract_feature(model, tsfm, os.path.join(faces_dir, folder, f), device)[0]

            # compair with all face
            scores = []
            labels = []
            for k, v in db.items():
                scores.append(cosin_metric(feature, v))
                labels.append(k)

            pred_score = max(scores)
            pred_label = labels[scores.index(pred_score)]
            if pred_score < 0.5:
                UNKNOW += 1  
            else:
                if pred_label == true_label:
                    RIGHT += 1
                else:
                    WRONG += 1
    
    prec_p = RIGHT / CNT
    miss_p = UNKNOW / CNT
    err_p = WRONG / CNT
    print('PREC-P: {}, MISS-P: {}, ERR-P: {}'.format(prec_p, miss_p, err_p))


def val_checkpoint(faces_dir, model, tsfm, device):
    same = defaultdict()
    diff = defaultdict()

    print('Val same face...')
    for folder in tqdm(os.listdir(faces_dir)):
        faces = os.listdir(os.path.join(faces_dir, folder))
        if len(faces) >= 2:
            f1 = extract_feature(model, tsfm, os.path.join(faces_dir, folder, faces[0]), device)[0]
            f2 = extract_feature(model, tsfm, os.path.join(faces_dir, folder, faces[1]), device)[0]
            cos = cosin_metric(f1, f2)
            same[folder] = cos

    print('Same face min vector: \n{}\n'.format(sorted(same.items(), key=lambda d:d[1], reverse = True)[-10:]))

    print('Val different face...')
    img_path = []
    for folder in os.listdir(faces_dir):
        faces = os.listdir(os.path.join(faces_dir, folder))
        if len(faces) >= 1:
            img_path.append(os.path.join(faces_dir, folder, faces[0]))

    for i in tqdm(range(1000)):
        img1 = random.choice(img_path)
        label1 = img1.split('/')[-2]
        img2 = random.choice(img_path)
        label2 = img2.split('/')[-2]

        if label1 == label2: continue
        
        f1 = extract_feature(model, tsfm, img1, device)[0]
        f2 = extract_feature(model, tsfm, img2, device)[0]
        cos = cosin_metric(f1, f2)
        diff[label1+'_'+label2] = cos

    print('Different face max vector: \n{}\n'.format(sorted(diff.items(), key=lambda d:d[1], reverse = True)[:10]))


def main():
    with open(args.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    project_name = data_dict['project_name']
    faces_dir = data_dict['faces_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = torch.load(args.checkpoint)['model'].module.to(device)
    else:
        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model'].module.to(device)
    model.eval()

    tsfm = T.Compose([T.Resize((112,112)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if args.which == 'db':
        val_db(faces_dir, model, tsfm, device)
    elif args.which == 'checkpoint':
        val_checkpoint(faces_dir, model, tsfm, device)


if __name__ == '__main__':
    main()
