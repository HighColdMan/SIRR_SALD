import cv2
import numpy as np
import random
from scipy.signal import convolve2d

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def syn(input1, input2):
    input1 = np.float32(input1) / 255.
    input2 = np.float32(input2) / 255.

    sigma = random.uniform(2, 5)
    R_blur = input2
    kernel = cv2.getGaussianKernel(11, sigma)
    kernel2d = np.dot(kernel, kernel.T)
    ###  H*W*C  C=3
    for i in range(3):
        R_blur[..., i] = convolve2d(R_blur[..., i], kernel2d, mode='same')
    M_ = input1 + R_blur
    if np.max(M_) > 1:
        m = M_[M_ > 1]
        m = (np.mean(m) - 1) * 1.3
        R_blur = np.clip(R_blur - m, 0, 1)
        M_ = np.clip(R_blur + input1, 0, 1)

    return np.float32(input1), np.float32(R_blur), np.float32(M_)

import numpy as np
from skimage import measure
# from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from functools import partial


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))       ### tile 平铺  延某一维度复制平铺
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.astype(imtype)
    return image_numpy


def ssim_access(X, Y):
    ssim = np.mean(cal_bwssim(Y, X))
    return ssim


def quality_assess(X, Y):
    # Y: correct; X: estimate
    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))
    res = {'PSNR': psnr, 'SSIM': ssim}
    return res, psnr, ssim
