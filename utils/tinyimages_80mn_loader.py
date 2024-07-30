import random

import torch
import torch.distributions
from torch.utils.data import Dataset
import numpy as np
import os
random.seed(1)

TINY_LENGTH = 79302017


def _load_cifar_exclusion_idcs(exclude_cifar, exclude_cifar10_1):
    cifar_idxs = []
    main_idcs_dir = 'datasets/unlabeled_datasets/80M_Tiny_Images/TinyImagesExclusionIdcs'

    our_exclusion_files = [
        '80mn_cifar10_test_idxs.txt',
        '80mn_cifar100_test_idxs.txt',
        '80mn_cifar10_train_idxs.txt',
        '80mn_cifar100_train_idxs.txt',
    ]
    if exclude_cifar:
        with open(os.path.join(main_idcs_dir, '80mn_cifar_idxs.txt'), 'r') as idxs:
            for idx in idxs:
                # indices in file take the 80mn database to start at 1, hence "- 1"
                cifar_idxs.append(int(idx) - 1)

        for file in our_exclusion_files:
            with open(os.path.join(main_idcs_dir, file), 'r') as idxs:
                for idx in idxs:
                    cifar_idxs.append(int(idx))

    if exclude_cifar10_1:
        with open(os.path.join(main_idcs_dir, '80mn_cifar101_idxs.txt'), 'r') as idxs:
            for idx in idxs:
                cifar_idxs.append(int(idx))

    cifar_idxs = torch.unique(torch.LongTensor(cifar_idxs))
    return cifar_idxs

# Code from https://github.com/hendrycks/outlier-exposure
class TinyImages(Dataset):
    def __init__(self, transform, exclude_cifar=True, exclude_cifar10_1=True, num_fixed_x=-1,
                 tiny_file='../../datasets/80M_Tiny_Images/tiny_images.bin'):
        self.memap = np.memmap(tiny_file, mode='r', dtype='uint8', order='C').reshape(TINY_LENGTH, -1)

        # if transform_base is not None:
        #     transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transform_base])
        # else:
        #     transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.ToTensor()])

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        exclusion_idcs = _load_cifar_exclusion_idcs(exclude_cifar, exclude_cifar10_1)
        if num_fixed_x > 0:
            inlusion_idcs = list(set(range(0, TINY_LENGTH)).difference(set(exclusion_idcs)))
            inlusion_idcs = random.sample(inlusion_idcs, num_fixed_x)
            exclusion_idcs = list(set(range(0, TINY_LENGTH)).difference(set(inlusion_idcs)))

        self.included_indices = torch.ones(TINY_LENGTH, dtype=torch.long)
        self.included_indices[exclusion_idcs] = 0
        self.included_indices = torch.nonzero(self.included_indices, as_tuple=False).squeeze()
        self.length = len(self.included_indices)
        print(f'80M Tiny Images - Length {self.length} - Excluding {len(exclusion_idcs)} images')

    def __getitem__(self, ii):
        index = self.included_indices[ii]
        img = self.memap[index].reshape(32, 32, 3, order="F")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return self.length


# Code from https://github.com/jkatzsam/woods_ood
class RandomImages(torch.utils.data.Dataset):

    def __init__(self, tiny_file='../../datasets/80M_Tiny_Images/300K_random_images.npy', transform=None, exclude_cifar=True):

        # tiny_file = np.load('/nobackup-slow/dataset/my_xfdu/300K_random_images.npy')
        data_file = np.load(tiny_file)

        def load_image(idx):
            # data_file.seek(idx * 3072)
            # data = data_file.read(3072)
            data = data_file[idx]
            return np.asarray(data, dtype='uint8')#.reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('utils/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs
            print(self.in_cifar)
            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search
        # print(f'300K RandomImages - Length {300000} - Excluding {300000 - len(self.in_cifar)} images')


    def __getitem__(self, index):
        index = (index + self.offset) % 299999

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(300000)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 300000


# if __name__ == '__main__':
#     # img_loader = ImageNet(id_type='cifar100', imagenet_dir='../../datasets/unlabeled_datasets/Imagenet64/')
#     import torchvision.transforms as T
#     out_transform_train = T.Compose([
#         T.ToTensor(),
#         T.ToPILImage(),
#         T.RandomCrop(32, padding=4),
#         T.RandomHorizontalFlip(),
#         T.ToTensor()
#     ])
#     out_kwargs = {}
#     train_ood_loader = torch.utils.data.DataLoader(
#         RandomImages(transform=out_transform_train, tiny_file='../../datasets/80M_Tiny_Images/300K_random_images.npy'),
#         batch_size=128, shuffle=False, **out_kwargs)
#     train_ood_iter = enumerate(train_ood_loader)
#     for i in range(0, 400):
#         _, (org_ood_x, org_ood_y) = next(train_ood_iter)
#         print(org_ood_y)
#         a = 0
