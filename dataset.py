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