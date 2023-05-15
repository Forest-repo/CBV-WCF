import os
import sys
import random
import json
import pickle
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
plt.rc('font',family='Times New Roman')

def read_split_data(root: str, tra_rate: float = 0.8, val_rate: float = 0.1, test_rate: float = 0.1):
    """
    Read the data set according to the specified division ratio
    """
    random.seed(42)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    object_class = [cla for cla in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, cla))]
    class_indices = dict((k, v) for v, k in enumerate(object_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all image paths of the training set
    train_images_label = []  # Store the index information of the labels corresponding to the images in the training set
    val_images_path = []  # Store all image paths of the validation set
    val_images_label = []  # Store the index information of the label corresponding to the verification set image
    test_images_path = []  # Store all image paths of the test set
    test_images_label = []  # Store the index information of the label corresponding to the test set image

    every_class_num = []  # Store the total number of samples for each class
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # Supported file extension types

    for cla in object_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in sorted(os.listdir(cla_path))
                  if os.path.splitext(i)[-1] in supported]
        # Get the index corresponding to this category
        image_class = class_indices[cla]
        # Record the number of samples for that category
        total = len(images)
        every_class_num.append(total)
        random.shuffle(images)
        train_images = images[0:int(tra_rate*total)]
        val_images = images[int(tra_rate*total):int((tra_rate+val_rate)*total)]
        test_images = images[int((tra_rate+val_rate)*total):]

        for img_path in images:
            if img_path in train_images:
                train_images_path.append(img_path)
                train_images_label.append(image_class)
            if img_path in val_images:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            if img_path in test_images:
                test_images_path.append(img_path)
                test_images_label.append(image_class)
        print(cla, ':', 'count:', total, ' train:', len(train_images),
              'val:', len(val_images), 'test:', len(test_images))

    print(
        f"There are {len(object_class)} class and {sum(every_class_num)} images were found in the dataset.")
    print("{} images for train.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    plot_image = True
    save_image = True
    if plot_image:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(object_class)), every_class_num, align='center')
        plt.xticks(range(len(object_class)),
                   object_class, fontsize=13, rotation=15)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 8, s=str(v), ha='center')
        plt.xlabel('Image class', fontdict={'fontsize': 17})
        plt.ylabel('Number of images', fontdict={'fontsize': 17})
        plt.title('Class distribution', fontdict={'fontsize': 18})
        plt.tight_layout()
        if save_image:
            plt.savefig("log_pictures/read_split_data.png",dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label


class custom_dataset(Dataset):
    """
    To customize a general-purpose single-view dataset,
    only the list of image paths and the list of labels are required as input.
    """
    def __init__(self, images_path: list, images_class: list, transforms=None):
        self.images_path = images_path
        self.images_class = images_class
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225])
            self.transforms = T.Compose(
                [T.Resize((224, 224)), T.RandomHorizontalFlip, T.ToTensor(), normalize])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        imgs_path = self.images_path[index]
        imgs_label = self.images_class[index]
        if imgs_path[-3:] in ['png', 'jpg', 'bmp']:
            img = Image.open(imgs_path)
            img = img.convert("RGB")
            img = self.transforms(img)
            label = imgs_label
        return img, label

    def __len__(self):
        return len(self.images_path)


class custom_dataset_pair(Dataset):
    """
    To customize the dual-view dataset,
    only the list of image paths and the list of labels are required as input
    """
    def __init__(self, aerial_images: list, aerial_class: list, ground_images: list, ground_class: list, transforms=None):

        self.aerial_imgs = aerial_images
        self.ground_imgs = ground_images
        self.aerial_class = aerial_class
        self.ground_class = ground_class
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225])
            self.transforms = T.Compose(
                [T.Resize((224, 224)), T.RandomHorizontalFlip, T.ToTensor(), normalize])
        else:
            self.transforms = transforms

    def __getitem__(self, index):

        aerial_imgs_path = self.aerial_imgs[index]
        aerial_imgs_label = self.aerial_class[index]
        ground_imgs_path = self.ground_imgs[index]
        ground_imgs_label = self.ground_class[index]
        if aerial_imgs_path[-3:] in ['png','jpg','bmp']:
            aerial_data = Image.open(aerial_imgs_path)
            aerial_data = aerial_data.convert("RGB")
            aerial_data = self.transforms(aerial_data)
        if ground_imgs_path[-3:] in ['png','jpg','bmp']:
            ground_data = Image.open(ground_imgs_path)
            ground_data = ground_data.convert("RGB")
            ground_data = self.transforms(ground_data)

        return [aerial_data,ground_data], aerial_imgs_label

    def __len__(self):
        return len(self.aerial_imgs)

def write_pickle(list_info: list, file_name: str):

    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:

    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def plot_data_loader_image(data_loader):
    """
    Plot the read dataset
    """
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Anti-Normalize operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()

def seed_it(seed):
    """
    Determine the random number seed
    """
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Adjust learning rate
    """
    lr = opt.learning_rate
    lr_max = opt.learning_rate
    warmup = opt.warmup
    warmup_epoch = 4 if warmup else 0

    if warmup:
        if epoch < warmup_epoch:
            lr = lr_max * epoch / warmup_epoch
        elif opt.cosine:
            eta_min = lr_max * (opt.lr_decay_rate ** 6)
            if warmup:
                lr = eta_min + (lr_max - eta_min) * (1 + math.cos(math.pi *(epoch - warmup_epoch) / (opt.epochs - warmup_epoch))) / 2
            else:
                lr = eta_min + (lr_max - eta_min) * (1 + math.cos(math.pi * epoch / opt.epochs)) / 2
        else:
            print('None')
    else:

        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            lr = lr * (opt.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class old_CBVLoss(nn.Module):
    """
    Consistency Between Views Loss
    old CBV_Loss
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(old_CBVLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CBVLoss_anchor_air(nn.Module):
    """
    Consistency Between Views Loss
    new CBV_Loss,
    Cross-view loss but the anchor only uses the air
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CBVLoss_anchor_air, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz],
            mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],' 'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.reshape(
                features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')

        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:

            labels = labels.reshape(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(
                device)
        else:
            mask = mask.float().to(device)

        anchor_feature = features[:, 0]

        contrast_feature = torch.cat(torch.unbind(
            features, dim=1), dim=0)
        contrast_feature = contrast_feature[batch_size:, ]

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(
            anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)

        # log_prob [20,20]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.reshape(1, batch_size).mean()

        return loss


class CBVLoss_anchor_both(nn.Module):
    """
    Consistency Between Views Loss
    Cross-view loss, the anchor uses the air and ground, pair matched by class label.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CBVLoss_anchor_both, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz],
            mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],' 'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.reshape(
                features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')

        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:

            labels = labels.reshape(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)  # bsz*bsz
        anchor_dot_contrast[0:batch_size,0:batch_size] = 0
        anchor_dot_contrast[batch_size:,batch_size:] = 0

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # .detach() remove gradient

        mask = mask.repeat(2, 2)
        mask[0:batch_size,0:batch_size] = 0
        mask[batch_size:,batch_size:] = 0

        # compute log_prob
        exp_logits = torch.exp(logits)

        # log_prob [20,20]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.reshape(2, batch_size).mean()

        return loss

class CBVLoss_anchor_both_random_pair(nn.Module):
    """
    Consistency Between Views Loss
    Cross-view loss, the anchor uses the air and ground, pair matched by random.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CBVLoss_anchor_both_random_pair, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels_view_1=None, labels_view_2=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz],
            mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.   返回一个损失标量
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],' 'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.reshape(
                features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels_view_1 is not None and labels_view_2 is None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')

        elif labels_view_1 is None and labels_view_2 is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels_view_1 is not None and labels_view_2 is not None:
            labels_view_1 = labels_view_1.reshape(-1, 1)
            labels_view_2 = labels_view_2.reshape(-1, 1)
            if labels_view_1.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels_view_1, labels_view_2.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        anchor_dot_contrast[0:batch_size,0:batch_size] = 0
        anchor_dot_contrast[batch_size:,batch_size:] = 0
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(2, 2)
        mask[0:batch_size,0:batch_size] = 0
        mask[batch_size:,batch_size:] = 0

        # compute log_prob
        exp_logits = torch.exp(logits)

        # log_prob [20,20]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.reshape(2, batch_size).mean()

        return loss

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def F1(matrix):
    """
    Calculate F1, precision, recall, etc.
    """
    sum_Pre = 0
    sum_Re = 0
    sum_all = 0
    sum_TP = 0
    for i in range(len(matrix)):
        TP = matrix[i][i]
        FP_and_TP = 0
        FN_and_TP = 0
        for j in range(len(matrix)):
            sum_all += matrix[i][j]
            FN_and_TP += matrix[i][j]
            FP_and_TP += matrix[j][i]
        sum_TP += TP
        if (FP_and_TP == 0):
            sum_Pre += 0.0
        else:
            sum_Pre += TP/FP_and_TP
        if FN_and_TP == 0:
            sum_Re += 0.0
        else:
            sum_Re += TP/FN_and_TP

    sum_Pre /= len(matrix)
    sum_Re /= len(matrix)

    return {'Precision': sum_Pre, 'Recall': sum_Re, 'F1': (2*sum_Pre*sum_Re)/(sum_Pre+sum_Re), 'Acc': sum_TP/sum_all}

def train(train_loader, model, criterion, optimizer, epoch):
    """
    one epoch training
    """
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)

        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT:{data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, idx + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
            print("now lr: ", optimizer.param_groups[0]['lr'])
            sys.stdout.flush()
    print('train: *** avg_Acc@1:{top1.avg:.3f} ***avg_losses:{losses.avg:.3f}'.format(
        top1=top1, losses=losses))

    return losses.avg, top1.avg

@torch.no_grad()
def validate(val_loader, model, criterion, epoch, num_class):
    """
    one epoch validation
    """
    model.eval()
    matrix = [[0 for j in range(num_class)]
              for i in range(num_class)]

    batch_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            pred = output.argmax(dim=1)
            pred = pred.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix[labels[id]][pred[id]] += 1
            result = F1(matrix)
            result['matrix '] = matrix

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print('validate: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                          idx + 1, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1, top5=top5))
    print('validate: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(
        top1=top1, result_F1=result['F1'], losses=losses))
    print(result)

    return losses.avg, top1.avg, matrix


@torch.no_grad()
def test(test_loader, model, criterion, num_class):
    """
    one epoch test
    """
    model.eval()
    matrix = [[0 for j in range(num_class)]
              for i in range(num_class)]  # Make a confusion matrix

    batch_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            pred = output.argmax(dim=1)
            pred = pred.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            for id in range(len(labels)):
                matrix[labels[id]][pred[id]] += 1
            result = F1(matrix)
            result['matrix '] = matrix

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print('validate: [{0}/{1}]\t'
                      'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1:{top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5:{top5.val:.3f} ({top5.avg:.3f})'.format(
                          idx + 1, len(test_loader), batch_time=batch_time,
                          loss=losses, top1=top1, top5=top5))
    print('validate: ***avg_Acc@1:{top1.avg:.3f} ***F1:{result_F1} ***avg_losses:{losses.avg:.3f}'.format(
        top1=top1, result_F1=result['F1'], losses=losses))
    print(result)

    return losses.avg, top1.avg, matrix

def matplot_loss(train_loss, val_loss, model_name, now_time):
    """
    Draw loss curve
    """
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("train_dataset and val_dataset loss")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(
        f"log_pictures/{now_time}/{model_name}/{model_name}_loss_curve_{now_time}.png")
    plt.show()
    plt.close()

def matplot_acc(train_acc, val_acc, model_name, now_time):
    """
    Draw acc curve
    """
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("train_dataset and val_dataset acc")
    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    plt.savefig(
        f"log_pictures/{now_time}/{model_name}/{model_name}_accuracy_curve_{now_time}.png")
    plt.show()
    plt.close()


def matplot_confusion_matrix(matrix, model_name, epoch, now_time, is_save=None):
    """
    Draw confusion matrix
    """
    # class label of Airound and  CvBRct
    classes = [{'airport': 0}, {'bridge': 1}, {'church': 2}, {'forest': 3}, {'lake': 4}, {
        'park': 5}, {'river': 6}, {'skyscraper': 7}, {'stadium': 8}, {'statue': 9}, {'tower': 10}]
    # classes=[{'apartment': 0}, {'hospital': 1}, {'house': 2}, {'industrial': 3}, {'parking_lot': 4},
    # {'religious': 5}, {'school': 6}, {'store': 7}, {'vacant_lot': 8}]

    classNamber = len(matrix)

    confusion_matrix = np.array(matrix)
    plt.figure(figsize=(15, 15))

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion_matrix', fontdict={'fontsize': 25})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, rotation=-45)
    plt.yticks(tick_marks, classes, fontsize=15)
    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(classNamber)]
                       for i in range(classNamber)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]), fontdict={
                 'fontsize': 20}, va='center', ha='center')  # show the corresponding numbers

    plt.ylabel('True label', fontdict={'fontsize': 20})
    plt.xlabel('Predict label', fontdict={'fontsize': 20})
    plt.tight_layout()

    if not os.path.isdir(f"log_pictures/{now_time}/{model_name}"):
        os.makedirs(f"log_pictures/{now_time}/{model_name}")
    if is_save is not None:
        plt.savefig(
            f"log_pictures/{now_time}/{model_name}/{model_name}_confusion_matrix{now_time}.png")
    plt.show()
    plt.close()
