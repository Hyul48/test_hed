from torch.utils import data
from os import listdir
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2


def prepare_image(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class BSDSLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb < 128] = 0.0
            lb[lb >= 128] = 1.0
        else:
            img_file = self.filelist[index].rstrip()
        img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
        img = prepare_image_cv2(img)
        if self.split == "train":
            return img, lb
        else:
            return img


class faultsDataset(data.Dataset):
    #     def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):

    def __init__(self, imgs_dir, masks_dir):
        #         self.train = train
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask = np.load("{}/{}.npy".format(self.masks_dir, idx))
        img = np.load("{}/{}.npy".format(self.images_dir, idx))
        #         mask_file = glob(self.masks_dir + idx + '.npy')
        #         img_file = glob(self.images_dir + idx + '.npy')

        #         assert len(mask_file) == 1, \
        #             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #         assert len(img_file) == 1, \
        #             f'Either no image or multiple images found for the ID {idx}: {img_file}'
        #         mask = np.load(mask_file[0])
        #         img = np.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return img, mask
