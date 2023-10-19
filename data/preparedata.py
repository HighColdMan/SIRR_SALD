
import os
import cv2
import torch
import random
import numpy as np

from ..utils import is_image_file
from .dataset import TrainDataset
from .dataset import TestDataset
# from SIRR.DATASET.dataset_224_288 import TrainDataset


random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


class DATASET:
    def __init__(self, train_args=None, test_args=None):
        self.train_args = train_args
        self.crop_size = self.train_args.crop_size
        self.max_shift_size = self.train_args.max_shift_size
        self.min_shift_size = self.train_args.min_shift_size

        self.test_args = test_args

    def make_train_dataset(self):
        dataset = []
        if self.train_args.syn_dir:
            input_syn_b, output_syn_t, output_syn_r = self.create_traindata_list([self.train_args.syn_dir],
                                                                                 is_ref_syn=True)
            train_syn_dataset = TrainDataset(input_syn_b, output_syn_t, output_syn_r, self.crop_size,
                                             self.min_shift_size, self.max_shift_size, is_ref_syn=True)
            train_dataloader_syn = torch.utils.data.DataLoader(train_syn_dataset,
                                                               batch_size=self.train_args.batch_size, shuffle=False,
                                                               num_workers=self.train_args.load_workers)
            dataset.append([train_dataloader_syn, True])

        if self.train_args.real_dir_89_200:
            input_real_b, output_real_t, output_real_r = self.create_traindata_list([self.train_args.real_dir_89_200],
                                                                                    is_ref_syn=False)
            train_real_dataset = TrainDataset(input_real_b, output_real_t, output_real_r, self.crop_size,
                                              self.min_shift_size, self.max_shift_size, is_ref_syn=False)
            train_dataloader_real89200 = torch.utils.data.DataLoader(train_real_dataset,
                                                                     batch_size=self.train_args.batch_size,
                                                                     shuffle=False,
                                                                     num_workers=self.train_args.load_workers,
                                                                     drop_last=True)
            dataset.append([train_dataloader_real89200, False])

        # if self.train_args.real_dir_unaligned250:
        #     input_real_b250, output_real_t250, output_real_r250 = self.create_traindata_list(
        #         [self.train_args.real_dir_unaligned250], is_ref_syn=False)
        #     train_real250_dataset = TrainDataset(input_real_b250, output_real_t250, output_real_r250, self.crop_size,
        #                                              self.min_shift_size, self.max_shift_size, is_ref_syn=False)
        #     train_dataloader_real250 = torch.utils.data.DataLoader(train_real250_dataset,
        #                                                            batch_size=self.train_args.batch_size,
        #                                                            shuffle=False,
        #                                                            num_workers=self.train_args.load_workers)
        #     dataset.append([train_dataloader_real250, False])

        # if self.train_args.real_dir_961:
        #     input_real_b961, output_real_t961, output_real_r961 = self.create_traindata_list([self.train_args.real_dir_961],
        #                                                                             is_ref_syn=False)
        #     train_real961_dataset = TrainDataset(input_real_b961, output_real_t961, output_real_r961, is_ref_syn=False)
        #     train_dataloader_real961 = torch.utils.data.DataLoader(train_real961_dataset,
        #                                                              batch_size=self.train_args.batch_size, shuffle=False,
        #                                                              num_workers=self.train_args.load_workers)
        #     dataset.append([train_dataloader_real961, False])

        # # ##dataset for refine
        # if self.train_args.real_dir_136:
        #     input_real_b136, output_real_t136, output_real_r136 = self.create_traindata_list(
        #         [self.train_args.real_dir_136], is_ref_syn=False)
        #     train_real136_dataset = TrainDataset(input_real_b136, output_real_t136, output_real_r136,
        #                                       is_ref_syn=False)
        #     train_dataloader_real136 = torch.utils.data.DataLoader(train_real136_dataset,
        #                                                            batch_size=self.train_args.batch_size, shuffle=False,
        #                                                            num_workers=self.train_args.load_workers)
        #     dataset.append([train_dataloader_real136, False])
        #
        # # ##dataset for refine
        # #
        return dataset

    def make_test_dataset(self):
        dataset = []
        if self.test_args.real20:
            image_list_real20, gt_list_real20 = self.creat_testdata_list(self.test_args.real20)
            test_dataset_real20 = TestDataset(image_list_real20, gt_list_real20, transform=True)
            test_loader_real20 = torch.utils.data.DataLoader(
                dataset=test_dataset_real20, batch_size=self.test_args.batch_size, shuffle=False, num_workers=self.test_args.num_workers)
            dataset.append([test_loader_real20, 'real20', True])
        if self.test_args.real45:
            image_list_real45, gt_list_real45 = self.creat_testdata_list(self.test_args.real45, if_gt=False)
            test_dataset_real45 = TestDataset(image_list_real45, gt_list_real45, transform=True, if_GT=False)
            test_loader_real45 = torch.utils.data.DataLoader(
                dataset=test_dataset_real45, batch_size=self.test_args.batch_size, shuffle=False, num_workers=self.test_args.num_workers)
            dataset.append([test_loader_real45, 'real45',  False])
        if self.test_args.WildSceneDataset:
            image_list_WildSceneDataset, gt_list_WildSceneDataset = self.creat_testdata_list(
                self.test_args.WildSceneDataset)
            test_dataset_WildSceneDataset = TestDataset(image_list_WildSceneDataset, gt_list_WildSceneDataset,
                                                        transform=True)
            test_loader_WildSceneDataset = torch.utils.data.DataLoader(
                dataset=test_dataset_WildSceneDataset, batch_size=self.test_args.batch_size, shuffle=False,
                num_workers=self.test_args.num_workers)
            dataset.append([test_loader_WildSceneDataset, 'WildSceneDataset', True])
        if self.test_args.SolidObjectDataset:
            image_list_SolidObjectDataset, gt_list_SolidObjectDataset = self.creat_testdata_list(
                self.test_args.SolidObjectDataset)
            test_dataset_SolidObjectDataset = TestDataset(image_list_SolidObjectDataset, gt_list_SolidObjectDataset,
                                                          transform=True)
            test_loader_SolidObjectDataset = torch.utils.data.DataLoader(
                dataset=test_dataset_SolidObjectDataset, batch_size=self.test_args.batch_size, shuffle=False,
                num_workers=self.test_args.num_workers)
            dataset.append([test_loader_SolidObjectDataset, 'SolidObjectDataset', True])

        if self.test_args.PostcardDataset:
            image_list_PostcardDataset, gt_list_PostcardDataset = self.creat_testdata_list(
                self.test_args.PostcardDataset)
            test_dataset_PostcardDataset = TestDataset(image_list_PostcardDataset, gt_list_PostcardDataset,
                                                       transform=True)
            test_loader_PostcardDataset = torch.utils.data.DataLoader(
                dataset=test_dataset_PostcardDataset, batch_size=self.test_args.batch_size, shuffle=False,
                num_workers=self.test_args.num_workers)
            dataset.append([test_loader_PostcardDataset, 'PostcardDataset', True])

        if self.test_args.syn_test_defocused:
            image_list_syn_test_defocused, gt_list_syn_test_defocused = self.creat_testdata_list(
                self.test_args.syn_test_defocused)
            test_dataset_syn_test_defocused = TestDataset(image_list_syn_test_defocused, gt_list_syn_test_defocused, transform=True)
            test_loader_syn_test_defocused = torch.utils.data.DataLoader(
                dataset=test_dataset_syn_test_defocused, batch_size=self.test_args.batch_size, shuffle=False,
                num_workers=self.test_args.num_workers)
            dataset.append([test_loader_syn_test_defocused, 'syn_test_defocused', True])

        # if self.test_args.syn_test_ghosting:
        #     image_list_syn_test_ghosting, gt_list_syn_test_ghosting = self.creat_testdata_list(
        #         self.test_args.syn_test_ghosting)
        #     test_dataset_syn_test_ghosting = TestDataset(image_list_syn_test_ghosting, gt_list_syn_test_ghosting, transform=True)
        #     test_loader_syn_test_ghosting = torch.utils.data.DataLoader(
        #         dataset=test_dataset_syn_test_ghosting, batch_size=self.test_args.batch_size, shuffle=False,
        #         num_workers=self.test_args.num_workers)
        #     dataset.append([test_loader_syn_test_ghosting, 'syn_test_ghosting', True])

        return dataset
    # ## for train dataset

    def create_traindata_list(self, train_path, is_ref_syn):
        blended = []
        transmission = []
        reflection = []
        for dirname in train_path:
            train_r_gt = dirname + "/reflection_layer/"
            train_t_gt = dirname + "/transmission_layer/"
            train_b = dirname + "/blended/"
            if is_ref_syn:
                r_list = os.listdir(train_r_gt)
                for _, _, fnames in sorted(os.walk(train_t_gt)):
                    for fname in fnames:
                        if is_ref_syn:
                            if fname not in r_list:
                                continue
                        if is_image_file(fname):
                            path_transmission = os.path.join(train_t_gt, fname)
                            path_reflection = os.path.join(train_r_gt, fname)
                            # path_blended = os.path.join(train_b, fname)
                            transmission_img = cv2.imread(path_transmission)
                            reflection_img = cv2.imread(path_reflection)
                            # blended_img = cv2.imread(path_blended)
                            transmission.append(transmission_img)
                            reflection.append(reflection_img)
                            # blended.append(blended_img)
            else:
                if os.path.exists(train_r_gt):
                    t_list = os.listdir(train_t_gt)
                    r_list = os.listdir(train_r_gt)
                    for _, _, fnames in sorted(os.walk(train_b)):
                        for fname in fnames:
                            if fname not in t_list and fname not in r_list:
                                continue
                            if is_image_file(fname):
                                path_blended = os.path.join(train_b, fname)
                                path_transmission = os.path.join(train_t_gt, fname)
                                path_reflection = os.path.join(train_r_gt, fname)
                                reflection_img = cv2.imread(path_reflection)
                                blended_img = cv2.imread(path_blended)
                                transmission_img = cv2.imread(path_transmission)
                                reflection.append(reflection_img)
                                transmission.append(transmission_img)
                                blended.append(blended_img)
                else:
                    t_list = os.listdir(train_t_gt)
                    for _, _, fnames in sorted(os.walk(train_b)):
                        for fname in fnames:
                            if fname not in t_list:
                                continue
                            if is_image_file(fname):
                                path_blended = os.path.join(train_b, fname)
                                path_transmission = os.path.join(train_t_gt, fname)
                                blended_img = cv2.imread(path_blended)
                                transmission_img = cv2.imread(path_transmission)
                                transmission.append(transmission_img)
                                blended.append(blended_img)
        return blended, transmission, reflection

    # ## for test dataset
    def creat_testdata_list(self, path, if_gt=True):
        gt_list = []
        image_list = []
        blended_path = path + '/blended/'

        if if_gt:
            trans_path = path + '/transmission_layer/'

            for _, _, fnames in sorted(os.walk(blended_path)):
                for fname in sorted(fnames):
                    path_image = os.path.join(blended_path, fname)
                    image = np.float32(cv2.imread(path_image)) / 255.0
                    image_list.append(image)

            for _, _, fnames in sorted(os.walk(trans_path)):
                for fname in sorted(fnames):
                    path_gt = os.path.join(trans_path, fname)
                    gt = np.float32(cv2.imread(path_gt)) / 255.0
                    gt_list.append(gt)

        else:
            for _, _, fnames in sorted(os.walk(path)):
                for fname in fnames:
                    path_image = os.path.join(path, fname)
                    image = np.float32(cv2.imread(path_image)) / 255.0
                    image_list.append(image)

        return image_list, gt_list
