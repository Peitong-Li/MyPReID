"""
@author: Lpt
@email: li2820447@163.com
@file: main.py
@time: 2021/4/21 14:53
@desc: the entrance of Project
"""
from __future__ import absolute_import
from Configs import args
import sys
import time
import datetime
import os.path as osp
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F

from Utils import data_manager
from Utils import transforms as T
from Utils.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLoss
from Utils.samplers import RandomIdentitySampler
from Utils.logger import Logger
from Utils.eval_metrics import evaluate
from Utils.optimizer import init_optim
from Utils.dataset_loader import ImageDataset
from Utils.util import AverageMeter, save_checkpoint
import Modules
from IPython import embed



def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False   # To save memory

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    # data augmentation
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
        # drop_last：batch_size 到最后可能有多余的几张图片不够一个bathc_size，舍掉这些图片
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,

    )

    print("Initializing model: {}".format(args.arch))
    model = Modules.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss=args.loss_type)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # if args.labelsmooth:
    #     criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    # else:
    #     criterion_class = CrossEntropyLoss(use_gpu=use_gpu)

    if args.loss_type == {'softmax', 'metric'}:
        criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
        criterion_metric = TripletLoss(margin=args.margin)
    elif args.loss_type == {'metric'}:
        criterion_metric = TripletLoss(margin=args.margin)
    elif args.loss_type == {'softmax'}:
        criterion_class = nn.CrossEntropyLoss()

    # criterion_metric = TripletLossAlignedReID(margin=args.margin)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    # Multiple GPUs run together
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")


    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()

        if args.loss_type == {'softmax', 'metric'}:
            train(epoch, model, optimizer, trainloader,criterion_class=criterion_class, criterion_metric=criterion_metric, use_gpu=use_gpu)
        elif args.loss_type == {'metric'}:
            train(epoch, model, optimizer, trainloader,criterion_metric=criterion_metric, use_gpu=use_gpu)
        elif args.loss_type == {'softmax'}:
            train(epoch, model, optimizer, trainloader,criterion_class=criterion_class, use_gpu=use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

# 可按照自己的模式进行修改
def train(epoch, model, optimizer, trainloader, criterion_class=None, criterion_metric=None, use_gpu=False):
    model.train()

    losses = AverageMeter()
    class_losses = AverageMeter()
    triplet_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        outputs, features = model(imgs)    # [batch_size, pids_num]
        class_loss = F.cross_entropy(outputs, pids)
        # class_loss = criterion_class(outputs, pids)
        triplet_loss = criterion_metric(features, pids)
        loss = class_loss + triplet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        class_losses.update(class_loss.item(), pids.size(0))
        triplet_losses.update(triplet_loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {class_loss.val:.4f} ({class_loss.avg:.4f})\t'
                  'MLoss {triplet_loss.val:.4f} ({triplet_loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time, data_time=data_time, loss=losses,
                   class_loss=class_losses, triplet_loss=triplet_losses))


def test(model, queryloader, galleryloader, use_gpu, ranks=[1,5,10,20]):
    batch_time = AverageMeter()

    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()

        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)

    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc[0]

if __name__ == '__main__':
    main()