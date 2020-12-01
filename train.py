import os
import argparse
import numpy as np
import torch
from torch import nn

from config import device, grad_clip
from utils.dataset import Dataset
from utils.utils import AverageMeter, accuracy, calc_num_person
from models.focal_loss import FocalLoss
from models.optimizer import InsightFaceOptimizer
from models.models import resnet18, resnet34, resnet50, resnet101, resnet152, MobileNet, resnet_face18, ArcMarginModel

parser = argparse.ArgumentParser(description='Train face network')
parser.add_argument('--train-path', default='./train-faces', help='train face path')
parser.add_argument('--epochs', type=int, default=100, help='training epoch size.')
parser.add_argument('--batch-size', type=int, default=64, help='batch size in each context')
parser.add_argument('--backbone', default='resnet101', help='specify backbone')
parser.add_argument('--resume-checkpoint', type=str, default=None, help='specify backbone')
parser.add_argument('--pretrained', type=bool, default=False, help='need pretrained model')
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='checkpoints')

parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
parser.add_argument('--optimizer', default='sgd', help='optimizer')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--mom', type=float, default=0.9, help='momentum')
parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
parser.add_argument('--focal-loss', type=bool, default=True, help='focal loss')
parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
parser.add_argument('--use-se', type=bool, default=True, help='use SEBlock')
parser.add_argument('--full-log', type=bool, default=False, help='full logging')
args = parser.parse_args()


def main():
    os.makedirs(args.checkpoints, exist_ok=True)
    torch.manual_seed(7)
    np.random.seed(7)
    best_acc = 0

    if args.resume_checkpoint is None:
        if args.backbone == 'r18':
            model = resnet18(args)
        elif args.backbone == 'r34':
            model = resnet34(args)
        elif args.backbone == 'r50':
            model = resnet50(args)
        elif args.backbone == 'resnet101':
            model = resnet101(args)
        elif args.backbone == 'r152':
            model = resnet152(args)
        elif args.backbone == 'mobile':
            model = MobileNet(1.0)
        else:
            model = resnet_face18(args.use_se)
        model = nn.DataParallel(model)
        num_person = calc_num_person(args.train_path)
        print('======================================')
        print('To be trained people number is: {}.'.format(num_person))
        print('======================================')
        metric_fc = ArcMarginModel(args, num_person)
        metric_fc = nn.DataParallel(metric_fc)

        if args.optimizer == 'sgd':
            optimizer = InsightFaceOptimizer(
                torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay))
        else:
            optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(args.resume_checkpoint)
        model = checkpoint['model']
        metric_fc = checkpoint['metric_fc']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = Dataset(root=args.train_path, phase='train', input_shape=(3, 112, 112))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(train_loader, model, metric_fc, criterion, optimizer, epoch)
        if acc > best_acc:
            status = {'epoch': epoch, 'acc': acc, 'model': model, 'metric_fc': metric_fc, 'optimizer': optimizer}
            torch.save(status, os.path.join(args.checkpoints, str(epoch) + '_' + '%.4f' % str(acc) + '.pth'))
            best_acc = acc


def train_one_epoch(train_loader, model, metric_fc, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1_accs = AverageMeter()

    model.train()
    metric_fc.train()

    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 10575]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top1_accuracy = accuracy(output, label, 1)
        top1_accs.update(top1_accuracy)

        # Print status
        print('Epoch: [{}][{}/{}], Loss {:.4f} ({:.4f}), Accuracy {:.3f} ({:.3f})'.format(epoch, i,
              len(train_loader), losses.val, losses.avg, top1_accs.val, top1_accs.avg))

    print('*** Current epoch LR: {}, Loss: {:.4f}, Accuracy: {:.4f}, Step_num: {} ***'.format(optimizer.lr,
          losses.avg, top1_accs.avg, optimizer.step_num))

    return losses.avg, top1_accs.avg


if __name__ == '__main__':
    main()
