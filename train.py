import os
import numpy as np
import argparse
import yaml
import torch
from torch import nn
from utils.d1 import Dataset
from utils.focal_loss import FocalLoss
from utils.models import resnet18, resnet34, resnet50, resnet101, resnet152, MobileNet, resnet_face18, ArcMarginModel
from utils.optimizer import InsightFaceOptimizer
#from utils import parse_args, save_checkpoint, AverageMeter, accuracy, get_logger
from utils.utils import save_checkpoint, AverageMeter, accuracy

parser = argparse.ArgumentParser(description='Train face network')
parser.add_argument('-p', '--project', default='./configs/ccfb.yaml', help='project file')
parser.add_argument('--backbone', default='r101', help='specify backbone')
parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')
parser.add_argument('--end-epoch', type=int, default=500, help='training epoch size.')
parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
parser.add_argument('--optimizer', default='sgd', help='optimizer')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--mom', type=float, default=0.9, help='momentum')
parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
parser.add_argument('--batch-size', type=int, default=128, help='batch size in each context')
parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
parser.add_argument('--focal-loss', type=bool, default=True, help='focal loss')
parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
parser.add_argument('--use-se', type=bool, default=True, help='use SEBlock')
parser.add_argument('--full-log', type=bool, default=False, help='full logging')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='checkpoint')
parser.add_argument('--resume', action='store_true', help='resume to train')
args = parser.parse_args()

def main():
    torch.manual_seed(7)
    np.random.seed(7)
    start_epoch = 0
    best_acc = 0
    epochs_since_improvement = 0
    
    with open(args.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    num_classes = data_dict['num_classes']
    grad_clip = data_dict['grad_clip']
    train_path = data_dict['train_path']
    project_name = data_dict['project_name']
    checkpoints_dir = os.path.join(args.checkpoints, project_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize / load checkpoint
    if not args.resume:
        if args.backbone == 'r18':
            model = resnet18(args)
        elif args.backbone == 'r34':
            model = resnet34(args)
        elif args.backbone == 'r50':
            model = resnet50(args)
        elif args.backbone == 'r101':
            model = resnet101(args)
        elif args.backbone == 'r152':
            model = resnet152(args)
        elif args.backbone == 'mobile':
            model = MobileNet(1.0)
        else:
            model = resnet_face18(args.use_se)
        model = nn.DataParallel(model)
        metric_fc = ArcMarginModel(device, num_classes, args)
        metric_fc = nn.DataParallel(metric_fc)

        if args.optimizer == 'sgd':
            optimizer = InsightFaceOptimizer(
                torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay))
        else:
            optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        metric_fc = checkpoint['metric_fc']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    #train_dataset = Dataset(root=train_path, phase='train', input_shape=(3, 112, 112))
    print(train_path)
    train_dataset = Dataset(root=train_path, input_shape=(3, 112, 112))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_top1_accs = train_one_epoch(train_loader=train_loader,
                                                      model=model,
                                                      metric_fc=metric_fc,
                                                      criterion=criterion,
                                                      optimizer=optimizer,
                                                      epoch=epoch, device=device, grad_clip=grad_clip)
        print('\nCurrent effective learning rate: {}\n'.format(optimizer.lr))
        print('Step num: {}\n'.format(optimizer.step_num))

        if train_top1_accs > best_acc:
            best_acc = train_top1_accs
            state = {'epoch': epoch,
                     'epochs_since_improvement': epochs_since_improvement,
                     'acc': 0,
                     'model': model,
                     'metric_fc': metric_fc,
                     'optimizer': optimizer}
            torch.save(state, os.path.join(checkpoints_dir, str(epoch) + '-' + str(best_acc) + '-' + args.backbone + '.pth'))


def train_one_epoch(train_loader, model, metric_fc, criterion, optimizer, epoch, device, grad_clip):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top1_accs = AverageMeter()

    # Batches
    #for i, (img, label, label_name) in enumerate(train_loader):
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        #print(label.shape)
        #print(label)
        #print(label_name)

        # Forward prop.
        #print(img.shape)
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 10575]
        #print(output.shape)

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

        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Top1 Accuracy {top1_accs.val:.3f} ({top1_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                               loss=losses,
                                                                               top1_accs=top1_accs))

    return losses.avg, top1_accs.avg


if __name__ == '__main__':
    main()
