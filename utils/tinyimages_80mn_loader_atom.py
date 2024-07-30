import os
import numpy as np
import torch

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=True, exclude_cifar10_1=True,
                 tiny_file='datasets/unlabeled_datasets/80M_Tiny_Images/tiny_images.bin', ):

        data_file = open(tiny_file, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        our_exclusion_files = [
            '80mn_cifar10_test_idxs.txt',
            '80mn_cifar100_test_idxs.txt',
            '80mn_cifar10_train_idxs.txt',
            '80mn_cifar100_train_idxs.txt',
        ]
        main_idcs_dir = 'datasets/unlabeled_datasets/80M_Tiny_Images/TinyImagesExclusionIdcs'

        self.cifar_idxs = []
        if exclude_cifar:
            with open(os.path.join(main_idcs_dir,'80mn_cifar_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)
            for file in our_exclusion_files:
                with open(os.path.join(main_idcs_dir, file), 'r') as idxs:
                    for idx in idxs:
                        self.cifar_idxs.append(int(idx))
        if exclude_cifar10_1:
            with open(os.path.join(main_idcs_dir, '80mn_cifar101_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                    self.cifar_idxs.append(int(idx))
            # hash table option
        self.cifar_idxs = set(self.cifar_idxs)
        self.in_cifar = lambda x: x in self.cifar_idxs

        print(f'80M Tiny Images - Length 79302017 - Excluding {len(self.cifar_idxs)} images')


    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 79302017


# Code from https://github.com/jkatzsam/woods_ood
class RandomImages(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, exclude_cifar=False):

        # data_file = np.load('/nobackup-slow/dataset/my_xfdu/300K_random_images.npy')
        data_file = np.load(root)

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

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search
        print(f'300K RandomImages - Length {300000} - Excluding {300000 - len(self.in_cifar)} images')


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

