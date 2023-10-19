import numpy as np
import torch
import os
import cv2
import torch.nn.functional as F
import torch.nn as nn

from model import model_sirr
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from .utils import tensor2im, quality_assess
import argparse
from collections import OrderedDict as odict
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def str2bool(v):
    return v.lower() in ('y', 'yes', 't', 'true', '1')
l1loss = nn.L1Loss()


def Crop_img(img):
    crop_w = img.shape[0] - 224
    crop_h = img.shape[1] - 224
    if crop_w == 0:
        random_w = 0
    else:
        random_w = int(np.random.randint(0, crop_w) / 2)
    if crop_h == 0:
        random_h = 0
    else:
        random_h = int(np.random.randint(0, crop_h) / 2)

    return img[random_w:random_w + 224, random_h:random_h + 224]

parser = argparse.ArgumentParser('test')
parser.add_argument('--save_result_path', type=str2bool, default=True, help="if save result")
parser.add_argument('--real20', type=str2bool, default=True, help="if real20 test")
parser.add_argument('--WildSceneDataset', type=str2bool, default=True, help="if WildSceneDataset test")
parser.add_argument('--SolidObjectDataset', type=str2bool, default=True, help="if SolidObjectDataset test")
parser.add_argument('--PostcardDataset', type=str2bool, default=True, help="if PostcardDataset test")
parser.add_argument('--real45', type=str2bool, default=True, help="if real45 test")
parser.add_argument('--syn_data', type=str2bool, default=True, help="if syn_data")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--batch_size', default=1, type=int, help="batch size")
args = parser.parse_args()

path_real45 = r'F:\pythonProject\SIRR\DATASET\testset\real45/'
# datasets with GT
path_real20 = r'F:\pythonProject\SIRR\DATASET\testset\real20'
path_sir_postcard = r'F:\pythonProject\SIRR\DATASET\testset\PostcardDataset'
path_sir_solid = r'F:\pythonProject\SIRR\DATASET\testset\SolidObjectDataset'
path_sir_wild = r'F:\pythonProject\SIRR\DATASET\testset\WildSceneDataset\withgt'
path_syn_defocused = r'F:\pythonProject\SIRR\DATASET\testset\syn_test_defocused'
path_syn_ghosting = r'F:\pythonProject\SIRR\DATASET\testset\syn_test_ghosting'

gt_test = model_sirr()
gt_test.cuda()
gt_test.eval()


def creat_list(path, if_gt=True):
    gt_list = []
    image_list = []

    if if_gt:
        blended_path = path + '/blended/'
        trans_path = path + '/transmission_layer/'
        for _, _, fnames in sorted(os.walk(blended_path)):
            for fname in sorted(fnames):
                image_list.append(blended_path + fname)

        for _, _, fnames in sorted(os.walk(trans_path)):
            for fname in sorted(fnames):
                gt_list.append(trans_path + fname)

    else:
        for _, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                image_list.append(path + fname)

    return image_list, gt_list


class TestDataset(Dataset):
    def __init__(self, blended_list, trans_list, transform=False, if_GT=True, crop=Crop_img):
        self.to_tensor = transforms.ToTensor()
        self.blended_list = blended_list
        self.trans_list = trans_list
        self.transform = transform
        self.if_GT = if_GT
        self.crop = Crop_img

    def __getitem__(self, index):
        blended = cv2.imread(self.blended_list[index])
        trans = blended
        if self.if_GT:
            trans = cv2.imread(self.trans_list[index])
        if self.transform == True:
            if trans.shape[0] > trans.shape[1]:
                neww = 300
                newh = round((neww / trans.shape[1]) * trans.shape[0])

            if trans.shape[0] < trans.shape[1]:
                newh = 300
                neww = round((newh / trans.shape[0]) * trans.shape[1])

            blended = cv2.resize(np.float32(blended), (neww, newh), cv2.INTER_CUBIC) / 255.0
            trans = cv2.resize(np.float32(trans), (neww, newh), cv2.INTER_CUBIC) / 255.0
            if neww % 8 != 0:
                neww = neww - (neww % 8)
            if newh % 8 != 0:
                newh = newh - (newh % 8)
            blended = blended[:newh, :neww]
            trans = trans[:newh, :neww]
        blended = self.to_tensor(blended)
        trans = self.to_tensor(trans)
        return blended, trans

    def __len__(self):
        return len(self.blended_list)


def test_diffdataset(test_loader, epoch, save_path=None, if_GT=True):
    ssim_sum = 0
    psnr_sum = 0
    time_sum = 0
    for j, (image, gt) in enumerate(test_loader):
        image = image.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            image.requires_grad_(False)

            time_start = time.time()
            output_t, output_r = gt_test(image)
            time_end = time.time()
            if j != 0:
                time_sum = time_sum + (time_end - time_start)
            print(time_end - time_start)

            output_r = tensor2im(output_r)
            output_t = tensor2im(output_t)
            image = tensor2im(image)
            gt = tensor2im(gt)

            if if_GT:
                res, psnr, ssim = quality_assess(output_t, gt)

                # print(res)
                ssim_sum += ssim
                psnr_sum += psnr

            if save_path:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                cv2.imwrite("%s/%s_b.png" % (save_path, j), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite("%s/%s_t.png" % (save_path, j), output_t, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite("%s/%s_r.png" % (save_path, j), output_r, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                if if_GT:
                    cv2.imwrite("%s/%s_gt.png" % (save_path, j), gt, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('time average:', time_sum/(len(test_loader)-1))
    print(len(test_loader) * args.batch_size, 'epoch:', epoch, 'SSIM:', ssim_sum / len(test_loader),
          'PSNR:', psnr_sum / len(test_loader)
          )

    return len(test_loader), ssim_sum, psnr_sum

def test_state(gt_state, epoch):
    # rmap_test.load_state_dict(rmap_state)
    # restormer_test.load_state_dict(restormer_state)
    #
    # del rmap_state
    # del restormer_state
    gt_test.load_state_dict(gt_state)
    del gt_state

    datasets = odict([('real20', True), ('WildSceneDataset', True), ('SolidObjectDataset', True), ('PostcardDataset', True), ('real45', False)])
    # datasets = odict([('real20', True)])
    # datasets = odict([('syn_data', True)])
    psnr_all, ssim_all, num_all = 0, 0, 0
    for dataset, with_GT in datasets.items():
        if getattr(args, dataset):
            if args.save_result_path:
                save_path = './result/epoch' + str(epoch) + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = save_path + dataset
            else:
                save_path = None
            print('testing dataset:', dataset)
            num, ssim_sum, psnr_sum = test_diffdataset(eval('test_loader_' + dataset), epoch, save_path,
                                                       if_GT=with_GT)
            if with_GT:
                psnr_all += psnr_sum
                ssim_all += ssim_sum
                num_all += num
    ssim_av = ssim_all / num_all
    psnr_av = psnr_all / num_all

    print('epoch:{} SSIM:{}  PSNR:{}'.format(epoch, ssim_av, psnr_av))
    return ssim_av, psnr_av


image_list_real20, gt_list_real20 = creat_list(path_real20)
test_dataset_real20 = TestDataset(image_list_real20, gt_list_real20, transform=True)
test_loader_real20 = torch.utils.data.DataLoader(dataset=test_dataset_real20,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

image_list_WildSceneDataset, gt_list_WildSceneDataset = creat_list(path_sir_wild)
test_dataset_WildSceneDataset = TestDataset(image_list_WildSceneDataset, gt_list_WildSceneDataset, transform=True)
test_loader_WildSceneDataset = torch.utils.data.DataLoader(dataset=test_dataset_WildSceneDataset,
                                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

image_list_PostcardDataset, gt_list_PostcardDataset = creat_list(path_sir_postcard)
test_dataset_PostcardDataset = TestDataset(image_list_PostcardDataset, gt_list_PostcardDataset, transform=True)
test_loader_PostcardDataset = torch.utils.data.DataLoader(dataset=test_dataset_PostcardDataset,
                                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

image_list_SolidObjectDataset, gt_list_SolidObjectDataset = creat_list(path_sir_solid)
test_dataset_SolidObjectDataset = TestDataset(image_list_SolidObjectDataset, gt_list_SolidObjectDataset, transform=True)
test_loader_SolidObjectDataset = torch.utils.data.DataLoader(dataset=test_dataset_SolidObjectDataset,
                                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

image_list_real45, gt_list_real45 = creat_list(path_real45, if_gt=False)
test_dataset_real45 = TestDataset(image_list_real45, gt_list_real45, transform=True, if_GT=False)
test_loader_real45 = torch.utils.data.DataLoader(dataset=test_dataset_real45,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# image_list_syn_data, gt_list_syn_data = creat_list(path_syn, if_gt=True)
# test_dataset_syn_data = TestDataset(image_list_syn_data, gt_list_syn_data, if_GT=True)
# test_loader_syn_data = torch.utils.data.DataLoader(dataset=test_dataset_syn_data,
#                                               batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


if __name__ == '__main__':
    checkpoint = r""
    for _, _, fnamelist in os.walk(checkpoint):
        for fname in fnamelist:
            if fname == 'epoch_070_P_23.583_S_0.866_G_0.617.pth':
                ckpt_path = os.path.join('./checkpoint/' + fname)
                ckpt_pre = torch.load(ckpt_path)
                print("loading checkpoint'{}'".format(ckpt_path))
                # psnr_av, ssim_av = test_state(ckpt_pre['netG_state'], 70)
                psnr_av, ssim_av = test_state(ckpt_pre['gt_state'], 72)

                print('The average PSNR/SSIM of all chosen testsets:', psnr_av, ssim_av)