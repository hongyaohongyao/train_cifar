import argparse
import json
import os
import random
import shutil

import numpy as np
import torch
import torchvision.models as models_torchvision  # networks from torchvision
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from tqdm import tqdm
from augmentation import mixup_train

import models  # our model
from log import get_logger


def get_lowcase_callable_dict(model_dict):
    return {
        name: v
        for name, v in model_dict.__dict__.items() if name.islower()
        and not name.startswith("__") and callable(model_dict.__dict__[name])
    }


model_domain = {
    "base": get_lowcase_callable_dict(models),
    "torchvision": get_lowcase_callable_dict(models_torchvision)
}

parser = argparse.ArgumentParser(description='PyTorch Image Training')
parser.add_argument('--name', default='mini_net_sgd', help='task name')
parser.add_argument('--gpu', default=None, type=str, help='GPU id to use.')

parser.add_argument('--data-root',
                    default="~/datasets/cifar10",
                    type=str,
                    help='GPU id to use.')

parser.add_argument('-a',
                    '--arch',
                    default='resnet18',
                    help='model architecture')
parser.add_argument('--domain',
                    default='base',
                    help='model domain: ' + '|'.join(model_domain.keys()))
parser.add_argument('--num-classes',
                    default=10,
                    type=int,
                    help='num of classes')
parser.add_argument(
    '-b',
    '--batch-size',
    default=128,
    type=int,
    help='mini-batch size (mini_net_sgd: 128), this is the total '
    'batch size of all GPUs on the current node when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (mini_net_sgd: 4)')
parser.add_argument('-e',
                    '--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained',
                    default='',
                    type=str,
                    help='path to pretrained model')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.01,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--optim',
                    type=str,
                    default='sgd',
                    choices=['sgd', 'adam'],
                    help='optimizer, default use sgd')
parser.add_argument(
    '--sche',
    default='multi_step',
    choices=['multi_step', 'cosine_annealing'],
    help='learning rate scheduler, default use mutistep lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=5e-4,
                    type=float,
                    help='weight decay (mini_net_sgd: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-m',
                    '--milestones',
                    type=int,
                    action='append',
                    help='milestones')
parser.add_argument(
    '--sche-t-max',
    default=0,
    type=int,
    help=
    't max for cosine annealing scheduler,non positive mean using epochs as tmax'
)
parser.add_argument('--sche-gamma',
                    default=0.1,
                    type=float,
                    help='gamme of scheduler')
parser.add_argument('--erase-aug',
                    action='store_true',
                    help='random erase augmentation')
parser.add_argument('--mixup-aug',
                    action='store_true',
                    help='mixup augmentation')
parser.add_argument('--mixup-alpha',
                    default=0.2,
                    type=float,
                    help='alpha of mixup augmentation')
parser.add_argument('--seed',
                    default=31,
                    type=int,
                    help='seed for initializing training. ')
args = parser.parse_args()
if args.gpu != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state, is_best, base_dir):
    filename = os.path.join(base_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(base_dir, 'model_best.pth.tar'))


def normal_train(net, inputs, targets, criterion, args, **kwargs):
    pred = net(inputs)
    loss = criterion(pred, targets)
    correct = torch.sum(
        torch.argmax(pred.data, dim=1) == targets) / targets.shape[0]
    return loss, correct


def train():
    best_acc = -1
    log_dir = f'run/{args.name}'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "env.json"), "w") as f:
        json.dump(vars(args), f)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    logger = get_logger(log_dir,
                        f'train.log',
                        resume=args.resume,
                        is_rank0=True)
    # training obj
    trans_list = [
        # transforms.RandomResizedCrop(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]
    if args.erase_aug:
        trans_list.append(transforms.RandomErasing())
    trans_train = transforms.Compose(trans_list)

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = CIFAR10(root=args.data_root,
                        train=True,
                        download=True,
                        transform=trans_train)
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   shuffle=True)

    test_set = CIFAR10(root=args.data_root,
                       train=False,
                       download=True,
                       transform=trans_test)
    test_loader = data.DataLoader(test_set,
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  shuffle=False)

    logger.info("=> creating model '{}',num of classes '{}'".format(
        args.arch, args.num_classes))
    try:
        model = model_domain[args.domain][args.arch](
            num_classes=args.num_classes)
    except KeyError:
        logger.error(
            f"model domain: {args.domain} include {'|'.join(model_domain[args.domain].keys())}"
        )

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    logger.info(f"using optimizer {optimizer.__class__.__name__}")
    criterion = nn.CrossEntropyLoss()
    if args.sche == 'cosine_annealing':
        if args.sche_t_max <= 0:
            args.sche_t_max = args.epochs
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.sche_t_max,
                                      verbose=True)
    else:
        scheduler = MultiStepLR(optimizer,
                                milestones=args.milestones,
                                gamma=args.sche_gamma,
                                verbose=True)
    logger.info(
        f"using learning rate scheduler {scheduler.__class__.__name__}")
    model = torch.nn.DataParallel(model)
    if args.pretrained:
        ###################
        #    预训练模型    #
        ###################
        if os.path.isfile(args.pretrained):
            logger.info("=> loading pretrained model '{}'".format(
                args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"=> loaded pretrained model {args.pretrained}")
        else:
            logger.info("=> no pretrained model found at '{}'".format(
                args.pretrained))
    elif args.resume:
        ###################
        #      中断重启    #
        ###################
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if args.mixup_aug:
        one_train = mixup_train
        logger.info(f"using mixup augmentation: alpha {args.mixup_alpha}")
    else:
        one_train = normal_train
    for epoch in range(args.start_epoch, args.epochs):
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        ###################
        #       训练      #
        ###################
        logger.info(f'Train/Epoch {epoch}/{args.epochs}')
        pbar = tqdm(train_loader,
                    desc="train",
                    total=len(train_set) // args.batch_size)
        epoch_loss = 0
        acc = 0
        batch_num = 0
        model.train()

        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.cuda()
            labels = labels.cuda()
            loss, correct = one_train(model, imgs, labels, criterion, args)
            # 更新梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 处理推理结果
            epoch_loss += loss.cpu().item()
            acc += correct
            batch_num += 1

        writer.add_scalar('Train/loss', epoch_loss / batch_num, epoch)
        writer.add_scalar('Train/acc', acc / batch_num, epoch)
        logger.info('Train/loss %.5f' % (epoch_loss / batch_num))
        logger.info('Train/acc %.5f' % (acc / batch_num))
        ###################
        #       评估       #
        ###################
        logger.info(f'Val/Epoch {epoch}/{args.epochs}')
        pbar = tqdm(test_loader,
                    desc="val",
                    total=len(test_set) // args.batch_size)
        epoch_loss = 0
        acc = 0
        batch_num = 0
        with torch.no_grad():
            model.eval()
            for i, (imgs, labels) in enumerate(pbar):
                imgs = imgs.cuda()
                labels = labels.cuda()
                pred = model(imgs)
                loss = criterion(pred, labels)
                # 处理推理结果
                epoch_loss += loss.cpu().item()
                acc += torch.sum(
                    torch.argmax(pred, dim=1) == labels) / labels.shape[0]
                batch_num += 1
            acc = acc / batch_num
            epoch_loss = epoch_loss / batch_num
            writer.add_scalar('Val/loss', epoch_loss, epoch)
            writer.add_scalar('Val/acc', acc, epoch)
            logger.info('Val/loss %.5f' % epoch_loss)
            logger.info('Val/acc %.5f' % acc)
        #更新学习率
        scheduler.step()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, log_dir)
        logger.info(f'Checkpoint {epoch} saved ! best: {is_best}')


if __name__ == '__main__':
    setup_seed(args.seed)
    train()
