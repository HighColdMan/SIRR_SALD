import os
import torch
from torch.utils.data import ConcatDataset
from .data.dataset import TrainDataset
from .utils import is_image_file
from .test import test_state
from loss_function import LDLLoss
from .model import model_sirr
from loss_function import VGGLoss
import cv2

EPS = 1e-12


class Solver:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 1
        # self.ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class

    def prepare_data(self, train_path, is_ref_syn, domain):
        blended = []
        transmission = []
        reflection = []
        if is_ref_syn:
            if domain == 'defocused':
                print('loading synthetic data(defocused type)...')
            if domain == 'focused':
                print('loading synthetic data(focused type)...')
            if domain == 'ghosting':
                print('loading synthetic data(ghosting type)...')
        else:
            print('loading real data...')

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
                            transmission_img = cv2.imread(path_transmission)
                            path_blended = os.path.join(train_b, fname)
                            blended_img = cv2.imread(path_blended)
                            path_reflection = os.path.join(train_r_gt, fname)
                            reflection_img = cv2.imread(path_reflection)
                            blended.append(blended_img)
                            transmission.append(transmission_img)
                            reflection.append(reflection_img)
            else:
                t_list = os.listdir(train_t_gt)
                for _, _, fnames in sorted(os.walk(train_b)):
                    for fname in fnames:
                        if fname not in t_list:
                            continue
                        if is_image_file(fname):
                            path_blended = os.path.join(train_b, fname)
                            blended_img = cv2.imread(path_blended)
                            blended.append(blended_img)
                            path_transmission = os.path.join(train_t_gt, fname)
                            transmission_img = cv2.imread(path_transmission)
                            transmission.append(transmission_img)
        return blended, transmission, reflection

    def l1_loss(self, input, output):
        return torch.mean(torch.abs(input - output))

    def l2_loss(self, input, output):
        return torch.mean(torch.square(input-output))

    def compute_grad(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

    def convert_L(self, img):
        img_L = (0.114 * img[0, 0, :, :] + 0.587 * img[0, 1, :, :] + \
                 0.299 * img[0, 2, :, :]).unsqueeze(0).unsqueeze(0)
        return img_L

    def train_model(self):
        self.vggloss = VGGLoss()
        # self.gt = GT()
        self.gt = model_sirr()
        self.gt.cuda()
        self.G_opt = torch.optim.Adam(self.gt.parameters(), lr=self.args.lr)
        # resume from a checkpoit
        if self.args.resume_file:
            if os.path.isfile(self.args.resume_file):
                print("loading checkpoint'{}'".format(self.args.resume_file))
                checkpoint = torch.load(self.args.resume_file)
                self.start_epoch = checkpoint['epoch'] + 1
                self.gt.load_state_dict(checkpoint['gt_state'])
                del (checkpoint)
                print("'{}' loaded".format(self.args.resume_file, self.args.start_epoch))
            else:
                print("no checkpoint found at '{}'".format(self.args.resume_file))
                return 1

        torch.backends.cudnn.benchmark = True

        input_real_b, output_real_t, output_real_r = self.prepare_data([self.args.ref_real_dir], is_ref_syn=False, domain="real89")
        input_syn_b, output_syn_t, output_syn_r = self.prepare_data([self.args.ref_syn_dir], is_ref_syn=True, domain="defocused")

        train_refreal_dataset = TrainDataset(input_real_b, output_real_t, output_real_r, is_ref_syn=False)
        train_dataloader_refreal = torch.utils.data.DataLoader(train_refreal_dataset,
                                                               batch_size=self.args.batch_size, shuffle=False,
                                                               num_workers=self.args.load_workers)
        train_refsyn_dataset = TrainDataset(input_syn_b, output_syn_t, output_syn_r, is_ref_syn=True)
        train_dataloader_refsyn = torch.utils.data.DataLoader(train_refsyn_dataset,
                                                              batch_size=self.args.batch_size, shuffle=False,
                                                              num_workers=self.args.load_workers)

        for epoch in range(self.start_epoch, self.args.num_epochs+1):
            G_loss_refsyn_avg, D_loss_refsyn_avg = self.train_epoch(train_dataloader_refsyn, epoch, is_ref_syn=True)
            G_loss_refreal_avg, D_loss_refreal_avg = self.train_epoch(train_dataloader_refreal, epoch, is_ref_syn=False)
            G_loss_avg = (G_loss_refreal_avg + G_loss_refsyn_avg) / 2
            D_loss_avg = (D_loss_refreal_avg + D_loss_refsyn_avg) / 2
            print('epoch:', epoch, 'G_loss:', G_loss_avg, 'D_loss:', D_loss_avg)
            if epoch % self.args.save_model_freq == 0:
                state = {
                    'epoch': epoch,
                    'gt_state': self.gt.state_dict()
                }

                ssim, psnr, = test_state(state['gt_state'], epoch)
                print('Saving checkpoint,epoch_{:0>3} G_loss:{} D_loss:{} P:{:.3f} S:{:.3f}'
                      .format(epoch, G_loss_avg, D_loss_avg, psnr, ssim))
                torch.save(state,
                           './checkpoint/epoch_{:0>3}_P_{:.3f}_S_{:.3f}_G_{:.3f}.pth'
                           .format(epoch, psnr, ssim, G_loss_avg, ))

    def train_epoch(self, train_dataloader_fusion, epoch, is_ref_syn):

        self.gt.train()
        G_loss_sum = 0
        D_loss_sum = 0

        for index, (input_b, gt_t, gt_r) in enumerate(train_dataloader_fusion):
            input_b = input_b.cuda(non_blocking=True)
            gt_t = gt_t.cuda(non_blocking=True)
            gt_r = gt_r.cuda(non_blocking=True)

            T, R = self.gt(input_b)
            self.G_opt.zero_grad()

            if is_ref_syn:
                FT_loss = self.vggloss(T, gt_t)
                Pixel_loss = self.l1_loss(T, gt_t)
                R_loss = self.l1_loss(R, gt_r)
                FR_loss = self.vggloss(R, gt_r)
                ldl_loss = LDLLoss(T, gt_t, 7)
                G_loss = 0.3 * FT_loss + Pixel_loss + R_loss + 0.2 * FR_loss + ldl_loss
                if index % self.args.print_freq == 0:
                    print('epoch:{0} num:{1}  G_loss:{2:.5f} '
                          'FT_loss:{3:.5f} Pixel_loss:{4:.5f} FR_loss:{5:.5f} R_loss:{6:.5f} ldl_loss:{7:.5f}'
                          .format(epoch, index, G_loss, 0.3 * FT_loss, Pixel_loss, 0.2 * FR_loss, R_loss, ldl_loss))
            else:
                FT_loss = self.vggloss(T, gt_t)
                Pixel_loss = self.l1_loss(T, gt_t)
                G_loss = 0.3 * FT_loss + Pixel_loss
                ldl_loss = LDLLoss(T, gt_t, 7)
                if index % self.args.print_freq == 0:
                    print('epoch:{0} num:{1}  G_loss:{2:.5f} '
                          'FT_loss:{3:.5f} Pixel_loss:{4:.5f} ldl_loss:{5:.5f}'
                          .format(epoch, index, G_loss, 0.3 * FT_loss, Pixel_loss, ldl_loss))
            G_loss_sum += G_loss.item()
            G_loss = G_loss.cuda()

            G_loss.backward()
            self.G_opt.step()
            torch.cuda.empty_cache()

        return G_loss_sum / index, D_loss_sum / index

