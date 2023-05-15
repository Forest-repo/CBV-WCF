import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from torchvision import transforms as T
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import argparse
import math
import warnings
#from torch.cuda.amp import autocast, GradScaler
from models import resnet18_Siamese_CBV_WCF_residual_learn_layer24_PT
from utils import read_split_data, custom_dataset_pair, CBVLoss_anchor_both, seed_it, adjust_learning_rate, AverageMeter, accuracy, F1
plt.rc('font',family='Times New Roman')

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,80,130',help='where to decay lr, can be a list,if is None,no lr_dacey!')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,help='decay for weight_decay')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--momentum', type=float, default=0.9,help='momentum')
    parser.add_argument('--cosine', action='store_true',help='using cosine annealing')
    parser.add_argument('--warmup', action='store_true',help='using warmup')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18_Siamese_CBV_WCF_residual_learn_layer24_PT')
    parser.add_argument('--dataset', type=str, default='Airound',
                        choices=['Airound_aerial','Cvbrct_aerial','Airound_ground','Cvbrct_ground','Cvbrct','Airound'], help='dataset')
    parser.add_argument('--is_test', default=None, type=str,help='is test true or false. ')

    opt = parser.parse_args()

    if opt.lr_decay_epochs is not None:
        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

    if opt.seed is not None:
        seed_it(opt.seed)
        warnings.warn('You have chosen to seed training.')

    if opt.dataset == 'Airound_aerial':
        opt.n_cls = 11
    elif opt.dataset == 'Airound_ground':
        opt.n_cls = 11
    elif opt.dataset == 'Cvbrct_aerial':
        opt.n_cls = 9
    elif opt.dataset == 'Cvbrct_ground':
        opt.n_cls = 9
    elif opt.dataset == 'Cvbrct':
        opt.n_cls = 9
    elif opt.dataset == 'Airound':
        opt.n_cls = 11
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def train(train_loader, model, criterion1, criterion2, criterion3, optimizer, epoch):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_1 = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()

    losses_2 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    losses_CVB = AverageMeter()
    losses_all = AverageMeter()

    #scaler = GradScaler()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images_view_1=images[0]
        images_view_2=images[1]
        images_view_1=torch.as_tensor(images_view_1)
        images_view_2=torch.as_tensor(images_view_2)
        images_view_1 = images_view_1.cuda(non_blocking=True)
        images_view_2 = images_view_2.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        optimizer.zero_grad()
        output1, output2, output1_contrast, output2_contrast = model(images_view_1,images_view_2,double_flag='True')
        loss1 = criterion1(output1, labels)
        loss2 = criterion2(output2, labels)
        features = torch.cat([output1_contrast.unsqueeze(1), output2_contrast.unsqueeze(1)], dim=1)
        loss_CVB = criterion3(features, labels)

        # update metric
        losses_1.update(loss1.item(), bsz)
        acc1, acc5 = accuracy(output1, labels, topk=(1, 5))
        top1_1.update(acc1[0], bsz)
        top5_1.update(acc5[0], bsz)

        losses_2.update(loss2.item(), bsz)
        acc1, acc5 = accuracy(output2, labels, topk=(1, 5))
        top1_2.update(acc1[0], bsz)
        top5_2.update(acc5[0], bsz)

        losses_CVB.update(loss_CVB.item(), bsz)

        #Loss_all = loss1 + loss2
        Loss_all = loss1 + loss2 + loss_CVB
        losses_all.update(Loss_all.item(), bsz)

        Loss_all.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('view_1_Train: [{0}][{1}/{2}]\t'
                  'BT:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT:{data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses_1, top1=top1_1, top5=top5_1))
            print("view_1_now lr: ", optimizer.param_groups[0]['lr'])
            print('view_2_Train: [{0}][{1}/{2}]\t'
                  'BT:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT:{data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses_2, top1=top1_2, top5=top5_2))
            print("view_2_now lr: ", optimizer.param_groups[0]['lr'])
            print("losses_CVB:{losses_CVB.val:.3f} ({losses_CVB.avg:.3f})".format(losses_CVB=losses_CVB))
            print("losses_all:{losses_all.val:.3f} ({losses_all.avg:.3f})".format(losses_all=losses_all))
            sys.stdout.flush()
    print('view_1_train: *** avg_Acc@1:{top1.avg:.3f} ***avg_losses:{losses.avg:.3f}'.format(top1=top1_1,losses=losses_1))
    print('view_2_train: *** avg_Acc@1:{top1.avg:.3f} ***avg_losses:{losses.avg:.3f}'.format(top1=top1_2,losses=losses_2))
    print("losses_all=loss1+loss2+loss_CVB: ***{losses_all.avg:.3f}".format(losses_all=losses_all))

    return losses_1.avg, top1_1.avg, losses_2.avg, top1_2.avg, losses_CVB.avg, losses_all.avg


# 定义一个验证函数, 里面有matrix的函数
def validate(val_loader, model, criterion1, criterion2,criterion3, epoch, num_class):
    """validation"""
    model.eval()
    matrix_1=[[0 for j in range(num_class)] for i in range(num_class)]
    matrix_2=[[0 for j in range(num_class)] for i in range(num_class)]
    batch_time = AverageMeter()

    losses_1 = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()

    losses_2 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    top1_SV1 = AverageMeter()
    top5_SV1 = AverageMeter()
    top1_SV2 = AverageMeter()
    top5_SV2 = AverageMeter()

    losses_CBV = AverageMeter()
    losses_all = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):

            images_view_1=images[0]
            images_view_2=images[1]

            images_view_1=torch.as_tensor(images_view_1)
            images_view_2=torch.as_tensor(images_view_2)

            images_view_1 = images_view_1.cuda(non_blocking=True)
            images_view_2 = images_view_2.cuda(non_blocking=True)

            labels = labels.cuda()
            bsz = labels.shape[0]

            # compute DV
            output1,output2,output1_contrast, output2_contrast = model(images_view_1,images_view_2,double_flag='True')
            loss1 = criterion1(output1, labels)
            loss2 = criterion2(output2, labels)

            # compute SV
            outputSV1,outputSV2,C,D = model(images_view_1,images_view_2,double_flag='False')

            # compute loss

            # update metric
            losses_1.update(loss1.item(), bsz)
            acc1, acc5 = accuracy(output1, labels, topk=(1, 5))
            top1_1.update(acc1[0], bsz)
            top5_1.update(acc5[0], bsz)

            losses_2.update(loss2.item(), bsz)
            acc1, acc5 = accuracy(output2, labels, topk=(1, 5))
            top1_2.update(acc1[0], bsz)
            top5_2.update(acc5[0], bsz)

            acc1, acc5 = accuracy(outputSV1, labels, topk=(1, 5))
            top1_SV1.update(acc1[0], bsz)
            top5_SV1.update(acc5[0], bsz)

            acc1, acc5 = accuracy(outputSV2, labels, topk=(1, 5))
            top1_SV2.update(acc1[0], bsz)
            top5_SV2.update(acc5[0], bsz)

            #CBV loss
            features = torch.cat([output1_contrast.unsqueeze(1), output2_contrast.unsqueeze(1)], dim=1)
            loss_CBV = criterion3(features, labels)

            losses_CBV.update(loss_CBV.item(), bsz)

            Loss_all = loss1 + loss2 + loss_CBV
            losses_all.update(Loss_all.item(), bsz)

            pred1=output1.argmax(dim=1)
            pred1 = pred1.cpu().numpy().tolist()
            labels=labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix_1[labels[id]][pred1[id]]+=1
            result_1=F1(matrix_1)
            result_1['matrix ']=matrix_1

            pred2=output2.argmax(dim=1)
            pred2 = pred2.cpu().numpy().tolist()
            #labels=labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix_2[labels[id]][pred2[id]]+=1
            result_2=F1(matrix_2)
            result_2['matrix ']=matrix_2

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print('view_1_validate: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses_1, top1=top1_1, top5=top5_1))
                print('view_2_validate: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses_2, top1=top1_2, top5=top5_2))
                print("losses_CBV:{losses_CBV.val:.3f} ({losses_CBV.avg:.3f})".format(losses_CBV=losses_CBV))
                print("losses_all:{losses_all.val:.3f} ({losses_all.avg:.3f})".format(losses_all=losses_all))
                print('Single_view view_1_validate: [{0}/{1}]\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(idx + 1, len(val_loader), top1=top1_SV1, top5=top5_SV1))
                print('Single_view view_2_validate: [{0}/{1}]\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(idx + 1, len(val_loader), top1=top1_SV2, top5=top5_SV2))

    print('view_1_test: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(top1=top1_1,result_F1=result_1['F1'],losses=losses_1)) 
    print('view_2_test: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(top1=top1_2,result_F1=result_2['F1'],losses=losses_2)) 
    print('Single_view view_1_test: ***avg_Acc@1:{top1.avg:.3f}'.format(top1=top1_SV1)) 
    print('Single_view view_2_test: ***avg_Acc@1:{top1.avg:.3f}'.format(top1=top1_SV2)) 

    print("losses_all=loss1+loss2+loss_CBV: ***{losses_all.avg:.3f}".format(losses_all=losses_all))

    print(result_1)
    print(result_2)

    return losses_1.avg, top1_1.avg ,matrix_1, losses_2.avg, top1_2.avg, matrix_2, losses_CBV.avg, losses_all.avg, top1_SV1.avg,top1_SV2.avg

def validate_latefusion_sum_product(val_loader, model, criterion, num_class):
    """validation,用于后融合,检验查看loss,但不更新网络"""
    model.eval()

    matrix_1=[[0 for j in range(num_class)] for i in range(num_class)]
    matrix_2=[[0 for j in range(num_class)] for i in range(num_class)]
    batch_time = AverageMeter()
    losses_sum = AverageMeter()
    losses_product = AverageMeter()
    top1_sum = AverageMeter()
    top5_sum = AverageMeter()
    top1_product = AverageMeter()
    top5_product = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images_view_1=images[0]
            images_view_2=images[1]

            images_view_1=torch.as_tensor(images_view_1)
            images_view_2=torch.as_tensor(images_view_2)

            images_view_1 = images_view_1.cuda(non_blocking=True)
            images_view_2 = images_view_2.cuda(non_blocking=True)
            #images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            #output1,output2 = model(images_view_1,images_view_2)
            output1, output2, output1_contrast, output2_contrast = model(images_view_1,images_view_2)
            output_1_sm=torch.nn.functional.softmax(output1,dim=1)
            output_2_sm=torch.nn.functional.softmax(output2,dim=1)

            output_sum = output_1_sm+output_2_sm  #+output_fusion_sm
            loss_sum = criterion(output_sum, labels)

            output_product = output_1_sm * output_2_sm   # output_fusion_sm
            loss_product = criterion(output_product, labels)

            # update metric
            losses_sum.update(loss_sum.item(), bsz)
            losses_product.update(loss_product.item(), bsz)
            acc1, acc5 = accuracy(output_sum, labels, topk=(1, 5))
            top1_sum.update(acc1[0], bsz)
            top5_sum.update(acc5[0], bsz)

            acc1, acc5 = accuracy(output_product, labels, topk=(1, 5))
            top1_product.update(acc1[0], bsz)
            top5_product.update(acc5[0], bsz)

            pred1=output_sum.argmax(dim=1)
            pred1 = pred1.cpu().numpy().tolist()
            labels=labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix_1[labels[id]][pred1[id]]+=1
            result_1=F1(matrix_1)
            result_1['matrix ']=matrix_1

            pred2=output_product.argmax(dim=1)
            pred2 = pred2.cpu().numpy().tolist()
            #labels=labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix_2[labels[id]][pred2[id]]+=1
            result_2=F1(matrix_2)
            result_2['matrix ']=matrix_2

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print('latefusion_sum_validate: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses_sum, top1=top1_sum, top5=top5_sum))

            if (idx + 1) % 10 == 0:
                print('latefusion_product_validate: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses_product, top1=top1_product, top5=top5_product))

    print('latefusion_sum_test: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(top1=top1_sum,result_F1=result_1['F1'],losses=losses_sum)) 
    print('latefusion_product_test: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(top1=top1_product,result_F1=result_2['F1'],losses=losses_product)) 

    print(result_1)
    print(result_2)

    return losses_sum.avg, top1_sum.avg, matrix_1, losses_product.avg, top1_product.avg, matrix_2

def test(test_loader, model, criterion, num_class):
    """one epoch test"""
    model.eval()
    matrix=[[0 for j in range(num_class)] for i in range(num_class)]
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            pred=output.argmax(dim=1)
            pred = pred.cpu().numpy().tolist()
            labels=labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix[labels[id]][pred[id]]+=1
            result=F1(matrix)
            result['matrix ']=matrix

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx + 1, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
    print('test: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(top1=top1,result_F1=result['F1'],losses=losses)) 
    print(result)
    return losses.avg, top1.avg ,matrix

def matplot_loss(train_loss, val_loss, model_name, now_time):
    """
    Draw the loss curve
    """
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='test_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("train_dataset and test_dataset loss")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(f"log_pictures/{now_time}/{model_name}/{model_name}_loss_curve_{now_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
def matplot_SVacc(val_acc, model_name, now_time):
    """
    Draw the SV acc curve
    """
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title("Single view val dataset acc")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(f"log_pictures/{now_time}/{model_name}/{model_name}_SVacc_{now_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def matplot_loss_sum_product(sum_loss, product_loss, model_name, now_time):
    """
    Draw the loss curve
    """
    plt.plot(sum_loss, label='sum_loss')
    plt.plot(product_loss, label='product_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("val :sum_loss and product loss ")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(f"log_pictures/{now_time}/{model_name}/{model_name}_loss_curve_{now_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def matplot_acc(train_acc, val_acc, model_name, now_time):
    """
    Draw the acc curve
    """
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("train_dataset and val_dataset acc")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(f"log_pictures/{now_time}/{model_name}/{model_name}_accuracy_curve_{now_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def matplot_acc_sum_product(sum_acc, product_acc, model_name, now_time):
    """
    Draw the acc curve
    """
    plt.plot(sum_acc, label='sum_acc')
    plt.plot(product_acc, label='product_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("val: sum_acc and product_acc")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(f"log_pictures/{now_time}/{model_name}/{model_name}_accuracy_curve_{now_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def matplot_confusion_matrix(dataset, matrix, model_name, epoch, now_time, is_save=None):
    """
    Draw the confusion_matrix
    """
    if dataset == 'Airound':
        classes = [{'airport': 0}, {'bridge': 1}, {'church': 2}, {'forest': 3}, {'lake': 4}, {'park': 5}, {'river': 6}, {'skyscraper': 7}, {'stadium': 8}, {'statue': 9}, {'tower': 10}]
    if dataset == 'Cvbrct':
        classes=[{'apartment': 0}, {'hospital': 1}, {'house': 2}, {'industrial': 3}, {'parking_lot': 4}, {'religious': 5}, {'school': 6}, {'store': 7}, {'vacant_lot': 8}]
    classNamber=len(matrix)

    confusion_matrix = np.array(matrix)
    plt.figure(figsize=(15,15))
    plt.imshow(confusion_matrix, interpolation='nearest' ,cmap=plt.cm.Blues)
    plt.title('confusion_matrix',fontdict={'fontsize':25})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, rotation=-45)
    plt.yticks(tick_marks, classes, fontsize=15)  #fontdict={'fontsize':20}
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i,j] for j in range(classNamber)] for i in range(classNamber)],(confusion_matrix.size,2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]),fontdict={'fontsize':20},va='center',ha='center')

    plt.ylabel('True label',fontdict={'fontsize':20})
    plt.xlabel('Predict label',fontdict={'fontsize':20})
    plt.tight_layout()

    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    if is_save is not None:
        plt.savefig(f"log_pictures/{now_time}/{model_name}/{model_name}_confusion_matrix{now_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    opt = parse_option()
    if torch.cuda.is_available():
        print('gpu个数:',torch.cuda.device_count())
        idx=torch.cuda.current_device()
        print('gpu名称:',torch.cuda.get_device_name(idx))
    print(opt)
    print("Data starts loading")
    normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_transform =T.Compose([
        T.RandomChoice([T.CenterCrop((500,500)),
                                 T.CenterCrop((400,400)),
                                 T.CenterCrop((300,300)),
                                 T.CenterCrop((224,224)),
                                ]),
        T.RandomRotation(15),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
        T.RandomHorizontalFlip(),
        T.Resize(opt.size),
        T.ToTensor(),
        normalize,
    ])
    val_transform = T.Compose([
        T.Resize(opt.size),
        T.ToTensor(),
        normalize])
    if opt.dataset == 'Cvbrct':
        opt.n_cls = 9
        root_aerial = f"./Datasets/CV_BrCT/aerial"
        root_ground = f"./Datasets/CV_BrCT/street"
        aerial_tra_path, aerial_tra_label, aerial_val_path, aerial_val_label, aerial_test_path, aerial_test_label = read_split_data(root_aerial, tra_rate=0.8, val_rate=0.2, test_rate=0)
        ground_tra_path, ground_tra_label, ground_val_path, ground_val_label, ground_test_path, ground_test_label = read_split_data(root_ground, tra_rate=0.8, val_rate=0.2, test_rate=0)
        train_dataset = custom_dataset_pair(aerial_tra_path, aerial_tra_label,ground_tra_path, ground_tra_label, transforms=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        val_dataset = custom_dataset_pair(aerial_val_path, aerial_val_label,ground_val_path, ground_val_label, transforms=train_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        print("Data have been loaded")

    if opt.dataset == 'Airound':
        opt.n_cls = 11
        root_aerial = f"./Datasets/AiRound/aerial"
        root_ground = f"./Datasets/AiRound/ground"
        aerial_tra_path, aerial_tra_label, aerial_val_path, aerial_val_label, aerial_test_path, aerial_test_label = read_split_data(root_aerial, tra_rate=0.8, val_rate=0.2, test_rate=0)
        ground_tra_path, ground_tra_label, ground_val_path, ground_val_label, ground_test_path, ground_test_label = read_split_data(root_ground, tra_rate=0.8, val_rate=0.2, test_rate=0)
        train_dataset = custom_dataset_pair(aerial_tra_path, aerial_tra_label,ground_tra_path, ground_tra_label, transforms=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        val_dataset = custom_dataset_pair(aerial_val_path, aerial_val_label,ground_val_path, ground_val_label, transforms=train_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        print("Data have been loaded")

    print("Model is loading")

    model = resnet18_Siamese_CIC_WCF_residual_learn_layer24_PT(num_classes=opt.n_cls, flag_share='False')
    print('model',model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model has loaded to {device}")

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    criterion_latefusion = nn.CrossEntropyLoss()
    criterion_CBV = CBVLoss_anchor_both(temperature=0.1)

    criterion_1 = criterion_1.cuda()
    criterion_2 = criterion_2.cuda()
    criterion_latefusion = criterion_latefusion.cuda()
    criterion_CBV = criterion_CBV.cuda()
    print(f"criterion_1 has loaded to {device}")
    print(f"criterion_2 has loaded to {device}")
    print(f"criterion_latefusion has loaded to {device}")
    print(f"criterion_CBV has loaded to {device}")

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            if opt.gpu is None:
                checkpoint = torch.load(opt.resume)
            else:
                loc = 'cuda:{}'.format(opt.gpu)
                checkpoint = torch.load(opt.resume, map_location=loc)
            opt.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            best_acc1=torch.tensor(best_acc1)
            if opt.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(opt.gpu)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    model_1_name=f"resnet18_Siamese_CIC_WCF_residual_learn_layer24_PT_1_{opt.dataset}_lr{opt.learning_rate}_bsz{opt.batch_size}_0.8tra_0.2test"
    model_2_name=f"resnet18_Siamese_CIC_WCF_residual_learn_layer24_PT_2_{opt.dataset}_lr{opt.learning_rate}_bsz{opt.batch_size}_0.8tra_0.2test"
    model_3_name=f"resnet18_Siamese_CIC_WCF_residual_learn_layer24_PT_latefusuion_{opt.dataset}_lr{opt.learning_rate}_bsz{opt.batch_size}_0.8tra_0.2test"
    model_4_name=f"resnet18_Siamese_CIC_WCF_residual_learn_layer24_PT_allmodel_{opt.dataset}_lr{opt.learning_rate}_bsz{opt.batch_size}_0.8tra_0.2test"

    now_time_second=time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
    now_time_day=time.strftime("%Y-%m-%d-", time.localtime())
    print('training on', device,'\n')
    print("=======================train=======================")


    train_loss_1 = []
    train_acc_1 = []
    val_loss_1 = []
    val_acc_1 = []
    val_acc_SV1 = []

    train_loss_2 = []
    train_acc_2 = []
    val_loss_2 = []
    val_acc_2 = []
    val_acc_SV2 = []

    #CBV loss
    train_loss_CBV = []
    val_loss_CBV = []
    train_loss_all = []
    val_loss_all = []

    val_acc_latefusion_sum = []
    val_acc_latefusion_product = []
    val_loss_latefusion_sum = []
    val_loss_latefusion_product = []

    epochs = opt.epochs
    best_acc1_1 = 0.0
    best_acc1_2 = 0.0
    best_acc1_SV1 = 0.0
    best_acc1_SV2 = 0.0
    best_acc1_1_epoch = 1
    best_acc1_2_epoch = 1
    best_acc1_SV1_epoch = 1
    best_acc1_SV2_epoch = 1

    min_loss_1 = 999
    min_loss_2 = 999
    min_loss_1_epoch = 1
    min_loss_2_epoch = 1
    best_acc1_sum = 0.0
    best_acc1_product = 0.0
    best_acc1_sum_epoch = 1
    best_acc1_product_epoch = 1

    for epoch in range(opt.start_epoch, epochs+1):
        adjust_learning_rate(opt, optimizer, epoch)
        start_train = time.time()
        training_loss1, training_acc1,training_loss2, training_acc2,training_loss_CBV, training_loss_all = train(train_dataloader, model, criterion_1, criterion_2, criterion_CBV, optimizer, epoch)
        valing_loss1,valing_acc1,matrix_view1,valing_loss2,valing_acc2,matrix_view2,valing_loss_CBV,valing_loss_all,valing_accSV1,valing_accSV2 = validate(val_dataloader, model, criterion_1, criterion_2,criterion_CBV, epoch, num_class=opt.n_cls)
        #air view
        training_acc1=training_acc1.cpu()
        valing_acc1= valing_acc1.cpu()
        train_loss_1.append(training_loss1)
        train_acc_1.append(training_acc1)
        val_loss_1.append(valing_loss1)
        val_acc_1.append(valing_acc1)
        valing_accSV1 = valing_accSV1.cpu()
        val_acc_SV1.append(valing_accSV1)
        #ground view
        training_acc2=training_acc2.cpu()
        valing_acc2= valing_acc2.cpu()
        train_loss_2.append(training_loss2)
        train_acc_2.append(training_acc2)
        val_loss_2.append(valing_loss2)
        val_acc_2.append(valing_acc2)
        valing_accSV2 = valing_accSV2.cpu()
        val_acc_SV2.append(valing_accSV2)
        train_loss_CBV.append(training_loss_CBV)
        val_loss_CBV.append(valing_loss_CBV)

        end_train =time.time()
        print('epoch {}, total time {:.2f}s'.format(epoch, end_train-start_train))
        start_train = time.time()
        sum_loss,sum_acc,matrix_sum,product_loss,product_acc,matrix_product=validate_latefusion_sum_product(val_dataloader,model,
        criterion_latefusion, num_class=opt.n_cls)
        sum_acc=sum_acc.cpu()
        product_acc=product_acc.cpu()
        val_acc_latefusion_sum.append(sum_acc)
        val_acc_latefusion_product.append(product_acc)
        val_loss_latefusion_sum.append(sum_loss)
        val_loss_latefusion_product.append(product_loss)
        end_train =time.time()

        # save view_air ckpt
        if epoch % opt.save_freq == 0:
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            save_file = os.path.join(folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            state = {
            'model': model.view1_net.state_dict(),
            'best_acc1':best_acc1_1,
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_1_name}'):
                os.mkdir(f'saved_ckpt_model/{model_1_name}')
            torch.save(state,f'saved_ckpt_model/{model_1_name}/{opt.model}_epoch_{epoch}.pth')
            print(f"save model, 第{epoch}轮epoch\n")
        # save view_ground ckpt
        if epoch % opt.save_freq == 0:
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            save_file = os.path.join(folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            state = {
            'model': model.view2_net.state_dict(),
            'best_acc1':best_acc1_2,
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_2_name}'):
                os.mkdir(f'saved_ckpt_model/{model_2_name}')
            torch.save(state,f'saved_ckpt_model/{model_2_name}/{opt.model}_epoch_{epoch}.pth')
            print(f"save model, 第{epoch}轮epoch\n")
        # Save the current overall model weight to facilitate visual tsne analysis at that time
        if epoch % opt.save_freq == 0:
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            save_file = os.path.join(folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            state = {
            'model': model.state_dict(),
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_4_name}'):
                os.mkdir(f'saved_ckpt_model/{model_4_name}')
            torch.save(state,f'saved_ckpt_model/{model_4_name}/{opt.model}_epoch_{epoch}.pth')
            print(f"save model, 第{epoch}轮epoch\n")

        if min_loss_1 > valing_loss1:
            min_loss_1 = valing_loss1
            min_loss_1_epoch = epoch
        if min_loss_2 > valing_loss2:
            min_loss_2 = valing_loss2
            min_loss_2_epoch = epoch
        if sum_acc > best_acc1_sum:
            best_acc1_sum = sum_acc
            best_acc1_sum_epoch = epoch
        if product_acc > best_acc1_product:
            best_acc1_product = product_acc
            best_acc1_product_epoch = epoch

        if valing_acc1 > best_acc1_1:
            best_acc1_1 = valing_acc1
            best_acc1_1_epoch = epoch
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            print(f"save now best model to saved_ckpt_model/{opt.model}_epoch_{epoch}_best\n")
            state = {
            'model': model.view1_net.state_dict(),
            'best_acc1':best_acc1_1,
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_1_name}'):
                os.mkdir(f'saved_ckpt_model/{model_1_name}')
            torch.save(state,f'saved_ckpt_model/{model_1_name}/{opt.model}_DV1best.pth')
            matplot_confusion_matrix(opt.dataset, matrix_view1, model_1_name, epoch, now_time=now_time_second, is_save=True)
            state = {
            'model': model.state_dict(),
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_4_name}'):
                os.mkdir(f'saved_ckpt_model/{model_4_name}')
            torch.save(state,f'saved_ckpt_model/{model_4_name}/{opt.model}_DV1bestNow.pth')

        if valing_acc2 > best_acc1_2:
            best_acc1_2 = valing_acc2
            best_acc1_2_epoch = epoch
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            print(f"save now best model to saved_ckpt_model/{opt.model}_epoch_{epoch}_best\n")
            state = {
            'model': model.view2_net.state_dict(),
            'best_acc1':best_acc1_2,
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_2_name}'):
                os.mkdir(f'saved_ckpt_model/{model_2_name}')
            torch.save(state,f'saved_ckpt_model/{model_2_name}/{opt.model}_DV2best.pth') 
            matplot_confusion_matrix(opt.dataset, matrix_view2, model_2_name, epoch, now_time=now_time_second, is_save=True)
            state = {
            'model': model.state_dict(),
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_4_name}'):
                os.mkdir(f'saved_ckpt_model/{model_4_name}')
            torch.save(state,f'saved_ckpt_model/{model_4_name}/{opt.model}_DV2bestNow.pth')
        if valing_accSV1 > best_acc1_SV1:
            best_acc1_SV1 = valing_accSV1
            best_acc1_SV1_epoch = epoch
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            print(f"save now best model to saved_ckpt_model/{opt.model}_epoch_{epoch}_best\n")
            state = {
            'model': model.view1_net.state_dict(),
            'best_acc1':best_acc1_SV1,
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_1_name}'):
                os.mkdir(f'saved_ckpt_model/{model_1_name}')
            torch.save(state,f'saved_ckpt_model/{model_1_name}/{opt.model}_SV1best.pth')
        if valing_accSV2 > best_acc1_SV2:
            best_acc1_SV2 = valing_accSV2
            best_acc1_SV2_epoch = epoch
            folder = 'saved_ckpt_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            print(f"save now best model to saved_ckpt_model/{opt.model}_epoch_{epoch}_best\n")
            state = {
            'model': model.view2_net.state_dict(),
            'best_acc1':best_acc1_SV2,
            'epoch': epoch,
            }
            if not os.path.exists(f'saved_ckpt_model/{model_2_name}'):
                os.mkdir(f'saved_ckpt_model/{model_2_name}')
            torch.save(state,f'saved_ckpt_model/{model_2_name}/{opt.model}_SV2best.pth')
        print('layer2 Learnable λa now value is:',model.fusion_layer2.pp.data)
        print('layer2 Learnable λg now value is:',model.fusion_layer2.qq.data)
        print('layer4 Learnable λa now value is:',model.fusion_layer4.pp.data)
        print('layer4 Learnable λg now value is:',model.fusion_layer4.qq.data)
        print('val_latefusion_sum_product_epoch {}, total time {:.2f}s'.format(epoch, end_train-start_train))
        print(f'view1: best_acc:{best_acc1_1},epoch:{best_acc1_1_epoch} min_loss:{min_loss_1},epoch:{min_loss_1_epoch}')
        print(f'view2: best_acc:{best_acc1_2},epoch:{best_acc1_2_epoch} min_loss:{min_loss_2},epoch:{min_loss_2_epoch}')
        print(f'best_acc1_sum: {best_acc1_sum},epoch:{best_acc1_sum_epoch}')
        print(f'best_acc1_product: {best_acc1_product},epoch:{best_acc1_product_epoch}')  
        print(f'viewSV1: best_acc:{best_acc1_SV1},epoch:{best_acc1_SV1_epoch}')
        print(f'viewSV2: best_acc:{best_acc1_SV2},epoch:{best_acc1_SV2_epoch}')

    # view-air
    print(f"save last model, 第{epochs}轮epoch\n")
    state = {
        'model': model.view1_net.state_dict(),
        #'optimizer': optimizer.state_dict(),
        'best_acc1':best_acc1_1,
        'epoch': epochs,
    }
    if not os.path.exists(f'saved_ckpt_model/{model_1_name}'):
        os.mkdir(f'saved_ckpt_model/{model_1_name}')
    torch.save(state,f'saved_ckpt_model/{model_1_name}/{opt.model}_epoch_{epochs}_last.pth')

    # view-ground
    print(f"save last model, 第{epochs}轮epoch\n")
    state = {
        'model': model.view2_net.state_dict(),
        #'optimizer': optimizer.state_dict(),
        'best_acc1':best_acc1_2,
        'epoch': epochs,
    }
    if not os.path.exists(f'saved_ckpt_model/{model_2_name}'):
        os.mkdir(f'saved_ckpt_model/{model_2_name}')
    torch.save(state,f'saved_ckpt_model/{model_2_name}/{opt.model}_epoch_{epochs}_last.pth')

    # save view-all ckpt by the last ckpt
    print(f"save last model, 第{epochs}轮epoch\n")
    state = {
        'model': model.state_dict(),
        'epoch': epochs,
    }
    if not os.path.exists(f'saved_ckpt_model/{model_4_name}'):
        os.mkdir(f'saved_ckpt_model/{model_4_name}')
    torch.save(state,f'saved_ckpt_model/{model_4_name}/{opt.model}_epoch_{epochs}_last.pth')

    matplot_loss(train_loss_1, val_loss_1, model_1_name, now_time_second)
    matplot_acc(train_acc_1, val_acc_1, model_1_name, now_time_second)
    matplot_SVacc(val_acc_SV1, model_1_name+'SV1', now_time_second)
    matplot_loss(train_loss_2, val_loss_2, model_2_name, now_time_second)
    matplot_acc(train_acc_2, val_acc_2, model_2_name, now_time_second)
    matplot_SVacc(val_acc_SV2, model_2_name+'SV2', now_time_second)
    matplot_confusion_matrix(opt.dataset, matrix_sum, model_3_name+'_matrix_sum', epoch, now_time=now_time_second, is_save=True)
    matplot_confusion_matrix(opt.dataset, matrix_product, model_3_name+'_matrix_product', epoch, now_time=now_time_second, is_save=True)

    matplot_loss_sum_product(val_loss_latefusion_sum, val_loss_latefusion_product, model_3_name+'_loss', now_time_second)
    matplot_acc_sum_product(val_acc_latefusion_sum, val_acc_latefusion_product, model_3_name+'_acc', now_time_second)

    print('=======================train and test finish=======================\n')
    print(f'view1 best val_acc:{best_acc1_1} epoch:{best_acc1_1_epoch}')
    print(f'view1 min_best val_loss:{min_loss_1} epoch:{min_loss_1_epoch}')
    print('===###===')
    print(f'view2 best val_acc:{best_acc1_2} epoch:{best_acc1_2_epoch}')
    print(f'view2 min_best val_loss:{min_loss_2} epoch:{min_loss_2_epoch}') 
    print('===###===')
    print(f'viewSV1: best_acc:{best_acc1_SV1},epoch:{best_acc1_SV1_epoch}')
    print(f'viewSV2: best_acc:{best_acc1_SV2},epoch:{best_acc1_SV2_epoch}')
    print('===###===')
    print(f'best_acc1_sum:{best_acc1_sum} epoch:{best_acc1_sum_epoch}')
    print(f'best_acc1_product:{best_acc1_product} epoch:{best_acc1_product_epoch}')

if __name__=="__main__":
    main()


