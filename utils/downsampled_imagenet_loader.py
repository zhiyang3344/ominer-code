import random

import numpy as np
import torch
import os
import pickle
import math

cifar10_excluded_03labels = [280, 133, 281, 264, 265, 266, 8, 268, 269, 270, 271, 272, 399, 274, 275, 276, 17, 278, 143, 408, 409, 148, 149, 151, 413, 414, 415, 416, 417, 34, 282, 283, 284, 285, 172, 174, 46, 49, 177, 10, 55, 11, 60, 140, 69, 70, 197, 200, 80, 84, 87, 89, 95, 99, 230, 231, 106, 499, 372, 116, 118, 500, 501, 243, 246, 279]
cifar10_excluded_035labels = [133, 265, 266, 268, 269, 270, 271, 272, 143, 274, 279, 280, 281, 282, 283, 172, 46, 60, 84, 230, 372, 500, 501]

cifar100_excluded_03labels = [1, 522, 523, 524, 13, 526, 23, 24, 28, 30, 543, 544, 34, 619, 44, 48, 50, 53, 58, 61, 67, 580, 76, 591, 592, 83, 85, 602, 603, 604, 605, 606, 607, 96, 608, 609, 101, 102, 103, 104, 614, 615, 617, 108, 621, 622, 623, 624, 625, 620, 618, 628, 630, 121, 634, 635, 129, 644, 645, 646, 650, 651, 652, 653, 655, 153, 157, 159, 163, 167, 680, 681, 682, 684, 685, 693, 694, 183, 698, 186, 188, 701, 190, 194, 195, 712, 713, 714, 205, 206, 209, 212, 216, 730, 218, 735, 224, 231, 746, 754, 249, 250, 251, 764, 765, 254, 255, 772, 260, 261, 264, 777, 275, 788, 276, 277, 278, 279, 280, 281, 282, 283, 286, 287, 288, 289, 291, 292, 299, 300, 819, 820, 307, 308, 309, 310, 312, 311, 317, 318, 831, 319, 320, 836, 838, 843, 849, 859, 863, 357, 358, 870, 361, 875, 363, 879, 882, 884, 374, 887, 890, 892, 385, 901, 914, 405, 920, 929, 939, 944, 442, 443, 958, 447, 959, 962, 451, 455, 972, 463, 464, 465, 978, 466, 467, 468, 469, 470, 471, 472, 473, 987, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 490, 491, 511]
cifar100_excluded_035labels =[1, 522, 523, 524, 13, 24, 28, 543, 44, 58, 61, 67, 76, 591, 85, 604, 605, 606, 607, 96, 609, 102, 615, 103, 617, 614, 108, 621, 622, 623, 624, 625, 628, 630, 121, 634, 645, 646, 650, 651, 653, 157, 159, 163, 680, 681, 682, 685, 188, 701, 190, 194, 195, 205, 209, 212, 216, 735, 746, 754, 250, 251, 254, 255, 777, 788, 281, 287, 289, 299, 307, 309, 310, 317, 319, 831, 843, 859, 361, 875, 882, 374, 901, 920, 944, 958, 959, 962, 475, 490, 511]

random.seed(1)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict



class ImageNetEXCIFAR(torch.utils.data.Dataset):
    cifar_idxs = []

    def __init__(self, transform=None, id_type='', exclude_cifar=True, excl_simi=0.35, num_included_classes=-1,
                 included_classes=[], num_fixed_x=-1, img_size=32, imagenet_dir='datasets/unlabeled_datasets/Imagenet64/'):
        self.S = np.zeros(11, dtype=np.int32)
        self.img_size = img_size
        self.labels = []
        self.imagenet_dir = imagenet_dir
        self.exclude_cifar = exclude_cifar
        self.excluded_cifar_flags = {}
        # assert id_type in ['cifar10', 'cifar100']
        num_total = 0
        num_total_excluded = 0
        self.included_classes = [] # start from 0
        if len(included_classes) > 0:
            self.included_classes = included_classes
        else:
            if id_type=='cifar10':
                if num_included_classes > 0:
                    if excl_simi == 0.35:
                        org_included_classes = list(set(range(1, 1001)).difference(set(cifar10_excluded_035labels)))
                    elif excl_simi == 0.3:
                        org_included_classes = list(set(range(1, 1001)).difference(set(cifar10_excluded_03labels)))
                    included_classes = random.sample(org_included_classes, num_included_classes)  # start from 1
                    self.included_classes = [i - 1 for i in included_classes]  # start from 0
                    exclude_classes = list(set(range(1, 1001)).difference(set(included_classes)))
                    cifar10_excluded_labels = exclude_classes  # start from 1
                else:
                    if excl_simi == 0.35:
                        cifar10_excluded_labels = cifar10_excluded_035labels
                    else:
                        cifar10_excluded_labels = cifar10_excluded_03labels
                    org_included_classes = list(set(range(1, 1001)).difference(set(cifar10_excluded_labels))) # start from 1
                    self.included_classes = [i - 1 for i in org_included_classes]  # start from 0

            elif id_type=='cifar100':
                if num_included_classes > 0:
                    if excl_simi == 0.35:
                        org_included_classes = list(set(range(1, 1001)).difference(set(cifar100_excluded_035labels)))
                    elif excl_simi == 0.3:
                        org_included_classes = list(set(range(1, 1001)).difference(set(cifar100_excluded_03labels)))
                    included_classes = random.sample(org_included_classes, num_included_classes) # start from 1
                    self.included_classes = [i - 1 for i in included_classes] # start from 0
                    exclude_classes = list(set(range(1, 1001)).difference(set(included_classes))) # start from 1
                    cifar100_excluded_labels = exclude_classes # start from 1
                else:
                    if excl_simi == 0.35:
                        cifar100_excluded_labels = cifar100_excluded_035labels
                    else:
                        cifar100_excluded_labels = cifar100_excluded_03labels
                    org_included_classes = list(set(range(1, 1001)).difference(set(cifar100_excluded_labels))) # start from 1
                    self.included_classes = [i - 1 for i in org_included_classes]  # start from 0
        # print('Included classes: {} (start from 1)'.format(included_classes))
        print('Returned included classes: {} (start from 0)'.format(self.included_classes))

        num_total = 0
        for idx in range(1, 11):
            data_file = os.path.join(self.imagenet_dir, 'train_data_batch_{}'.format(idx))
            d = unpickle(data_file)
            y = d['labels']

            selected_y = []
            batch_excluded = np.zeros_like(y) != 0

            if not exclude_cifar or id_type not in ['cifar10', 'cifar100']:
                y = [i - 1 for i in y]
                selected_y = y
                print("WARNING, exclude_cifar is False or id_type not in ['cifar10', 'cifar100'], I will not exclude any OOD sample!")
            else:
                included_idx = []
                for i in range(0, len(y)):
                    single_y = y[i]
                    if id_type == 'cifar10':
                        if single_y in cifar10_excluded_labels:
                            batch_excluded[i] = True
                        else:
                            included_idx.append(i)
                    elif id_type == 'cifar100':
                        if single_y in cifar100_excluded_labels:
                            batch_excluded[i] = True
                        else:
                            included_idx.append(i)
                temp_y = np.array(y) - 1 # shift it so that indexes start at 0
                selected_y = temp_y[np.logical_not(batch_excluded)]

            if num_fixed_x > 0:
                print("len(included_idx)", len(included_idx), 'num_fixed_x', num_fixed_x)
                fixed_idx = random.sample(included_idx, math.ceil(num_fixed_x / 10))
                not_fixed_idx = list(set(included_idx).difference(set(fixed_idx)))
                for nf_idx in not_fixed_idx:
                    batch_excluded[nf_idx] = True
                temp_y = np.array(y) - 1  # shift it so that indexes start at 0
                selected_y = temp_y[np.logical_not(batch_excluded)]

            self.labels.extend(selected_y)
            self.S[idx] = self.S[idx - 1] + len(selected_y)
            self.excluded_cifar_flags[idx] = batch_excluded
            print('loaded train_data_batch_{}, total number:{}, selected samples:{}, excluded samples:{}'
                  .format(idx, len(y), len(selected_y), batch_excluded.sum()))
            num_total += len(y)
            num_total_excluded += batch_excluded.sum()
        self.labels = np.array(self.labels)
        self.N = len(self.labels)
        self.curr_batch = -1
        print('Total number:{}, total selected samples:{}, total excluded samples:{}, selected label set:{} (len: {})'
                .format(num_total, self.N, num_total_excluded, set(self.labels), len(set(self.labels))))
        self.offset = 0     # offset index
        self.transform = transform

    def get_selected_classes(self):
        return self.included_classes # start from 0

    def load_image_batch(self, batch_index):
        data_file = os.path.join(self.imagenet_dir, 'train_data_batch_{}'.format(batch_index))
        d = unpickle(data_file)
        x = d['data']
        
        img_size = self.img_size
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))

        if not self.exclude_cifar:
            self.batch_images = x
        else:
            included_cifar = np.logical_not(self.excluded_cifar_flags[batch_index])
            # print('included_cifar.sum()', included_cifar.sum())
            self.batch_images = x[included_cifar]
        self.curr_batch = batch_index

    def get_batch_index(self, index):
        j = 1
        while index >= self.S[j]:
            j += 1
        return j

    def load_image(self, index):
        batch_index = self.get_batch_index(index)
        if self.curr_batch != batch_index:
            self.load_image_batch(batch_index)
        
        return self.batch_images[index-self.S[batch_index-1]]

    def __getitem__(self, index):
        index = (index + self.offset) % self.N
        # print('index:', index)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index] # start from 0

    def __len__(self):
        return self.N


    def exlude_imagenet(self):
        from nltk.corpus import wordnet as wn
        id_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # id_labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'fish', 'flatfish', 'ray', 'shark', 'trout',
        #              'orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'bottle', 'bowl', 'can', 'cup', 'plate', 'apple',
        #              'mushroom', 'orange', 'pear', 'sweet_pepper', 'clock', 'computer_keyboard', 'lamp', 'telephone',
        #              'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly',
        #              'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle',
        #              'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle',
        #              'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab',
        #              'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile',
        #              'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        #              'maple', 'oak', 'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
        #              'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
        excluded_ood_ind = set()
        for raw_id_label in id_labels:
            temp = raw_id_label + '.n.01'
            id_label = wn.synset(raw_id_label + '.n.01')
            with open('map_clsloc.txt', 'r') as idxs:
                labels = []
                for line in idxs:
                    line = line.rstrip('\n')
                    row_label = line.split(' ')[0]
                    row_ind = line.split(' ')[1]
                    label = line.split(' ')[2]
                    ood_label = wn.synset(label + '.n.01')
                    simi = id_label.path_similarity(ood_label)
                    if simi >= 0.35:
                        print(raw_id_label, row_label, row_ind, label, simi)
                        excluded_ood_ind.add(int(row_ind))


# if __name__ == '__main__':
#     # img_loader = ImageNet(id_type='cifar100', imagenet_dir='../../datasets/unlabeled_datasets/Imagenet64/')
#     import torchvision.transforms as T
#     out_transform_train = T.Compose([
#         T.ToTensor(),
#         T.ToPILImage(),
#         # T.Pad(4, padding_mode='reflect'),
#         # T.RandomCrop(32),
#         T.RandomCrop(32, padding=4),
#         T.RandomHorizontalFlip(),
#         T.ToTensor()
#     ])
#     out_kwargs = {}
#     train_ood_loader = torch.utils.data.DataLoader(
#         ImageNetEXCIFAR(transform=out_transform_train, id_type='cifar10', num_included_classes=4,
#                         imagenet_dir='../../datasets/Imagenet32_train/'), batch_size=128, shuffle=False,
#         **out_kwargs)
#     train_ood_iter = enumerate(train_ood_loader)
#     for i in range(0, 20):
#         _, (org_ood_x, org_ood_y) = next(train_ood_iter)
#         print(org_ood_y)
#         a=0