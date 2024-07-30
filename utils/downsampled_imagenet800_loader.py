import math

import numpy as np
import torch
import os
import pickle
import random

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


wnids_800 = set()
current_path = os.path.dirname(__file__)
with open(os.path.join(current_path, './800_wnids.txt')) as f:
    for line in f:
        line = line.rstrip('\n')
        wnids_800.add(line)
ind_to_rowy = dict()
rowy_to_ind = dict()
with open(os.path.join(current_path, './map_clsloc.txt')) as f:
    for line in f:
        line = line.rstrip('\n')
        row_label, row_ind, label = line.split(' ')
        ind_to_rowy[int(row_ind)] = row_label
        rowy_to_ind[row_label] = int(row_ind)


class ImageNet800(torch.utils.data.Dataset):
    def __init__(self, transform=None, exclude_tiny200=True, img_size=64, num_included_classes=-1, included_classes=[],
                 num_fixed_x=-1, imagenet_dir='datasets/unlabeled_datasets/Imagenet64/'):
        self.len_batch = np.zeros(11, dtype=np.int32)
        self.img_size = img_size
        self.labels = []
        self.imagenet_dir = imagenet_dir
        self.exclude_tiny200 = exclude_tiny200
        self.excluded_tiny200_flags = {}
        # assert id_type in ['cifar10', 'cifar100']
        num_total = 0
        num_total_excluded = 0

        self.selected_wnids = set()
        if len(included_classes) > 0:
            self.selected_wnids = set(included_classes)
        elif num_included_classes > 0:
            self.selected_wnids = random.sample(wnids_800, num_included_classes)
        else:
            pass
        selected_ys = []
        for rowy in self.selected_wnids:
            selected_y = rowy_to_ind[rowy]
            selected_ys.append(selected_y-1)
        print('Info, length of selected_ys: {}, including {}'.format(len(selected_ys), selected_ys))

        for idx in range(1, 11):
            data_file = os.path.join(self.imagenet_dir, 'train_data_batch_{}'.format(idx))
            d = unpickle(data_file)
            y = d['labels']

            selected_y = []
            batch_excluded = np.zeros_like(y) != 0
            if not exclude_tiny200:
                # Labels are indexed from 1, shift it so that indexes start at 0
                y = [i - 1 for i in y]
                selected_y = y
                print("WARNING, exclude_tiny200 is False, I will not exclude any OOD sample!")
            else:
                included_idx = []
                for i in range(0, len(y)):
                    single_y = y[i]
                    if len(self.selected_wnids) == 0:
                        if ind_to_rowy[single_y] not in wnids_800:
                            batch_excluded[i] = True
                        else:
                            # shift it so that indexes start at 0
                            selected_y.append(single_y - 1)
                            included_idx.append(i)
                    else:
                        if ind_to_rowy[single_y] not in self.selected_wnids:
                            batch_excluded[i] = True
                        else:
                            selected_y.append(single_y - 1)
                            included_idx.append(i)

            if num_fixed_x > 0:
                print("len(included_idx)", len(included_idx), 'num_fixed_x', num_fixed_x)
                fixed_idx = random.sample(included_idx, math.ceil(num_fixed_x / 10))
                not_fixed_idx = list(set(included_idx).difference(set(fixed_idx)))
                for nf_idx in not_fixed_idx:
                    batch_excluded[nf_idx] = True
                temp_y = np.array(y) - 1  # shift it so that indexes start at 0
                selected_y = temp_y[np.logical_not(batch_excluded)]

            self.labels.extend(selected_y)
            self.len_batch[idx] = self.len_batch[idx-1] + len(selected_y)
            self.excluded_tiny200_flags[idx] = batch_excluded
            print('loaded train_data_batch_{}, total number:{}, selected samples:{}, excluded samples:{}'
                  .format(idx, len(y), len(selected_y), batch_excluded.sum()))
            num_total += len(y)
            num_total_excluded += batch_excluded.sum()
        self.labels = np.array(self.labels)
        self.N = len(self.labels)
        self.curr_batch = -1
        print('Total number:{}, total selected samples:{}, total excluded samples:{}'
              .format(num_total, self.N, num_total_excluded))
        self.offset = 0  # offset index
        self.transform = transform

    def load_image_batch(self, batch_index):
        data_file = os.path.join(self.imagenet_dir, 'train_data_batch_{}'.format(batch_index))
        d = unpickle(data_file)
        x = d['data']

        img_size = self.img_size
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))

        if not self.exclude_tiny200:
            self.batch_images = x
        else:
            included_img_inds = np.logical_not(self.excluded_tiny200_flags[batch_index])
            # print('included_cifar.sum()', included_cifar.sum())
            self.batch_images = x[included_img_inds]
        self.curr_batch = batch_index

    def get_batch_index(self, index):
        j = 1
        while index >= self.len_batch[j]:
            j += 1
        return j

    def load_image(self, index):
        batch_index = self.get_batch_index(index)
        if self.curr_batch != batch_index:
            self.load_image_batch(batch_index)
        return self.batch_images[index - self.len_batch[batch_index - 1]]

    def __getitem__(self, index):
        index = (index + self.offset) % self.N
        # print('index:', index)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

    def __len__(self):
        return self.N

    def get_selected_classes(self):
        selected_ys = []
        if len(self.selected_wnids) > 0:
            for rowy in self.selected_wnids:
                selected_y = rowy_to_ind[rowy] - 1
                selected_ys.append(selected_y)
        else:
            for rowy in wnids_800:
                selected_y = rowy_to_ind[rowy]
                selected_ys.append(selected_y - 1)
        return selected_ys  # start from 0

# if __name__ == '__main__':
#
#     # img_loader = ImageNet(id_type='cifar100', imagenet_dir='../../datasets/unlabeled_datasets/Imagenet64/')
#     import torchvision.transforms as T
#
#     out_transform_train = T.Compose([
#         T.ToTensor(),
#         T.ToPILImage(),
#         T.RandomCrop(32, padding=4),
#         # T.RandomCrop(64, padding=8),
#         T.RandomHorizontalFlip(),
#         T.ToTensor()
#     ])
#     out_kwargs = {}
#     train_ood_loader = torch.utils.data.DataLoader(
#         ImageNetRC(transform=out_transform_train, imagenet_dir='../../datasets/Imagenet32_train/', img_size=32),
#         batch_size=128, shuffle=False, **out_kwargs)
#     train_ood_iter = enumerate(train_ood_loader)
#     for i in range(0, 200):
#         _, (org_ood_x, org_ood_y) = next(train_ood_iter)
#         print(org_ood_y.min(), org_ood_y.max())

