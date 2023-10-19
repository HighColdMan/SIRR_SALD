import cv2
import math

import random
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from scipy.signal import convolve2d
from .synmethod import syn_defocused, syn_ghosting


# #

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


to_tensor = transforms.ToTensor()


class TrainDataset(Dataset):

    def __init__(self, dir_b_list, dir_t_list, dir_r_list, crop_size=224, is_ref_syn=True):

        self.dir_b_list = dir_b_list
        self.dir_t_list = dir_t_list
        self.dir_r_list = dir_r_list
        self.is_ref_syn = is_ref_syn
        self.crop_size = crop_size

        self.i = 0

    def __getitem__(self, index):

        t_img = self.dir_t_list[index]

        if self.is_ref_syn:
            r_img = self.dir_r_list[index]

            oh_t = t_img.shape[0]
            ow_t = t_img.shape[1]
            oh_r = r_img.shape[0]
            ow_r = r_img.shape[1]
            new = int(random.randint(self.crop_size, int(self.crop_size*1.2)))       # 1.5
            neww_t = round((new / t_img.shape[0]) * t_img.shape[1])
            newh_t = round((new / t_img.shape[1]) * t_img.shape[0])
            neww_r = round((new / r_img.shape[0]) * r_img.shape[1])
            newh_r = round((new / r_img.shape[1]) * r_img.shape[0])
            if ow_t >= oh_t:
                t_img = cv2.resize(np.float32(t_img), (neww_t, new), cv2.INTER_CUBIC)
            if oh_t > ow_t:
                t_img = cv2.resize(np.float32(t_img), (new, newh_t), cv2.INTER_CUBIC)
            if ow_r >= oh_r:
                r_img = cv2.resize(np.float32(r_img), (neww_r, new), cv2.INTER_CUBIC)
            if oh_r > ow_r:
                r_img = cv2.resize(np.float32(r_img), (new, newh_r), cv2.INTER_CUBIC)
            # # 训练数据为 defocused 反射类型
            t_img, r_img = self.crop(t_img, r_img, syn=True)

            t_img, r_img, b_img = syn_defocused(t_img, r_img)

        else:
            if len(self.dir_r_list) != 0:
                b_img = self.dir_b_list[index]
                r_img = self.dir_r_list[index]
                oh = t_img.shape[0]
                ow = t_img.shape[1]
                new = int(random.randint(self.crop_size, int(self.crop_size * 1.5)))
                neww = round((new / t_img.shape[0]) * t_img.shape[1])
                newh = round((new / t_img.shape[1]) * t_img.shape[0])
                if ow >= oh:
                    t_img = cv2.resize(np.float32(t_img), (neww, new), cv2.INTER_CUBIC) / 255.0
                    b_img = cv2.resize(np.float32(b_img), (neww, new), cv2.INTER_CUBIC) / 255.0
                    r_img = cv2.resize(np.float32(r_img), (neww, new), cv2.INTER_CUBIC) / 255.0

                    b_img, t_img, r_img = self.crop(b_img, t_img, r_img, syn=False)
                if oh > ow:
                    t_img = cv2.resize(np.float32(t_img), (new, newh), cv2.INTER_CUBIC) / 255.0
                    b_img = cv2.resize(np.float32(b_img), (new, newh), cv2.INTER_CUBIC) / 255.0
                    r_img = cv2.resize(np.float32(r_img), (new, newh), cv2.INTER_CUBIC) / 255.0
                    b_img, t_img, r_img = self.crop(b_img, t_img, img3=r_img, syn=False)
            else:
                b_img = self.dir_b_list[index]
                oh = t_img.shape[0]
                ow = t_img.shape[1]
                new = int(random.randint(self.crop_size, int(self.crop_size*1.5)))
                neww = round((new / t_img.shape[0]) * t_img.shape[1])
                newh = round((new / t_img.shape[1]) * t_img.shape[0])
                if ow >= oh:
                    t_img = cv2.resize(np.float32(t_img), (neww, new), cv2.INTER_CUBIC) / 255.0
                    b_img = cv2.resize(np.float32(b_img), (neww, new), cv2.INTER_CUBIC) / 255.0
                    # if new == self.crop_size:
                    #     randh = 0
                    # else:
                    #     randh = int(random.randint(0, new - self.crop_size))
                    # if neww == self.crop_size:
                    #     randw = 0
                    # else:
                    #     randw = int(random.randint(0, neww - self.crop_size))
                    # t_img = t_img_[randh:randh + self.crop_size, randw:randw + self.crop_size]
                    # b_img = b_img_[randh:randh + self.crop_size, randw:randw + self.crop_size]

                if oh > ow:
                    t_img = cv2.resize(np.float32(t_img), (new, newh), cv2.INTER_CUBIC) / 255.0
                    b_img = cv2.resize(np.float32(b_img), (new, newh), cv2.INTER_CUBIC) / 255.0
                    # if new == self.crop_size:
                    #     randw = 0
                    # else:
                    #     randw = int(random.randint(0, new - self.crop_size))
                    # if newh == self.crop_size:
                    #     randh = 0
                    # else:
                    #     randh = int(random.randint(0, newh - self.crop_size))
                    # t_img = t_img_[randh:randh + self.crop_size, randw:randw + self.crop_size]
                    # b_img = b_img_[randh:randh + self.crop_size, randw:randw + self.crop_size]

                b_img, t_img = self.crop(b_img, t_img, syn=False)
                r_img = cv2.subtract(b_img, t_img)
        b_img = to_tensor(b_img)
        r_img = to_tensor(r_img)
        t_img = to_tensor(t_img)

        return b_img, t_img, r_img

    def __len__(self):
        return len(self.dir_t_list)

    def crop(self, img1, img2, img3=None, syn=False):
        # 将图片Crop成大小为[224,224]
        if syn:
            crop_w1 = img1.shape[0] - self.crop_size
            crop_h1 = img1.shape[1] - self.crop_size
            rand_w1 = int(random.randint(0, crop_w1)) if crop_w1 != 0 else 0
            rand_h1 = int(random.randint(0, crop_h1)) if crop_h1 != 0 else 0
            crop_w2 = img2.shape[0] - self.crop_size
            crop_h2 = img2.shape[1] - self.crop_size
            rand_w2 = int(random.randint(0, crop_w2)) if crop_w2 != 0 else 0
            rand_h2 = int(random.randint(0, crop_h2)) if crop_h2 != 0 else 0

            img1 = img1[rand_w1:rand_w1 + self.crop_size, rand_h1:rand_h1 + self.crop_size]
            img2 = img2[rand_w2:rand_w2 + self.crop_size, rand_h2:rand_h2 + self.crop_size]

            return img1, img2
        else:
            if img3 is None:
                crop_w = img1.shape[0] - self.crop_size
                crop_h = img1.shape[1] - self.crop_size
                rand_w = int(random.randint(0, crop_w)) if crop_w != 0 else 0
                rand_h = int(random.randint(0, crop_h)) if crop_h != 0 else 0
                img1 = img1[rand_w:rand_w + self.crop_size, rand_h:rand_h + self.crop_size]
                img2 = img2[rand_w:rand_w + self.crop_size, rand_h:rand_h + self.crop_size]
                return img1, img2
            else:
                crop_w = img1.shape[0] - self.crop_size
                crop_h = img1.shape[1] - self.crop_size
                rand_w = int(random.randint(0, crop_w)) if crop_w != 0 else 0
                rand_h = int(random.randint(0, crop_h)) if crop_h != 0 else 0
                img1 = img1[rand_w:rand_w + self.crop_size, rand_h:rand_h + self.crop_size]
                img2 = img2[rand_w:rand_w + self.crop_size, rand_h:rand_h + self.crop_size]
                img3 = img3[rand_w:rand_w + self.crop_size, rand_h:rand_h + self.crop_size]
                return img1, img2, img3


class TestDataset(Dataset):
    def __init__(self, blended_list, trans_list, transform=False, if_GT=True):
        self.blended_list = blended_list
        self.trans_list = trans_list
        self.transform = transform
        self.if_GT = if_GT

    def __getitem__(self, index):
        blended = self.blended_list[index]
        oh, ow = blended.shape[0], blended.shape[1]
        if self.if_GT:
            trans = self.trans_list[index]
        else:
            trans = blended
        if self.transform:
            if oh > ow:
                blended = cv2.resize(np.float32(blended), (304, int(oh / ow * 304)), cv2.INTER_CUBIC)
                trans = cv2.resize(np.float32(trans), (304, int(oh / ow * 304)), cv2.INTER_CUBIC)
                blended = blended[int(oh / ow * 304) % 16:, :]
                trans = trans[int(oh / ow * 304) % 16:, :]
            else:
                blended = cv2.resize(np.float32(blended), (int(ow / oh * 304), 304), cv2.INTER_CUBIC)
                trans = cv2.resize(np.float32(trans), (int(ow / oh * 304), 304), cv2.INTER_CUBIC)
                blended = blended[:, int(ow / oh * 304) % 16:]
                trans = trans[:, int(ow / oh * 304) % 16:]
        # blended, trans = self.crop(blended, trans, crop_size=256)
        # cv2.imshow('test_blended', blended)
        # cv2.waitKey()
        # cv2.imshow('test_gt', trans)
        # cv2.waitKey()
        # blended, trans = self.crop(blended, trans)

        blended = to_tensor(blended)
        trans = to_tensor(trans)

        return blended, trans

    def __len__(self):
        return len(self.blended_list)

    def crop(self, blended, trans, crop_size=256):
        crop_w = blended.shape[0] - crop_size
        crop_h = blended.shape[1] - crop_size
        rand_w = int(random.randint(0, crop_w)) if crop_w != 0 else 0

        rand_h = int(random.randint(0, crop_h)) if crop_h != 0 else 0

        blended = blended[rand_w:rand_w + crop_size, rand_h:rand_h + crop_size]
        trans = trans[rand_w:rand_w + crop_size, rand_h:rand_h + crop_size]
        return blended, trans

