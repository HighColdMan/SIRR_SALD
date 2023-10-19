
import cv2
import numpy as np
from numpy import random
from scipy.signal import convolve2d

random_seed = 1234
np.random.seed(random_seed)
random.seed(random_seed)


def syn_defocused(transmission_img, reflection_img):
    transmission_img = np.float32(transmission_img) / 255.0
    reflection_img = np.float32(reflection_img) / 255.0
    sigma = random.uniform(2, 5)
    R_blur = reflection_img
    kernel = cv2.getGaussianKernel(11, sigma)
    kernel2d = np.dot(kernel, kernel.T)
    for i in range(3):
        R_blur[..., i] = convolve2d(R_blur[..., i], kernel2d, mode='same')
    blended = transmission_img + R_blur
    if np.max(blended) > 1:
        m = blended[blended > 1]
        m = (np.mean(m) - 1) * 1.3
        R_blur = np.clip(R_blur - m, 0, 1)
        blended = np.clip(R_blur + transmission_img, 0, 1)
    return transmission_img, R_blur, blended


def syn_focused(transmission, reflection):
    w = 0.1
    # a = 0.2 * np.random.random() + 0.8    # 0.7 不错
    a = float(0.2 * np.random.random() + 0.8)  # 0.7 不错
    b = float(0.2 * np.random.random() + 0.2)
    kernel = np.full_like(np.ones((5, 5)), w)
    for j in range(0, 3):
        reflection[..., j] = convolve2d(reflection[..., j], kernel, mode='same')

    blended = a * transmission + b * reflection
    return transmission, reflection, blended


def clamp_number(num, a, b):
    return max(min(num, max(a, b)), min(a, b))


def syn_ghosting(transmission, reflection, min_shift_size, max_shift_size):
    transmission = np.float32(transmission) / 255.0
    reflection = np.float32(reflection) / 255.0

    # ###添加高斯模糊
    sigma = random.uniform(2, 5)
    # kernel_size = 11
    # kernel_size = random.randint(3, 7)
    kernel_size = random.randint(1, 3) * 2 + 1
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel2d = np.dot(kernel, kernel.T)
    for j in range(3):
        reflection[..., j] = convolve2d(reflection[..., j], kernel2d, mode='same')
    # ###添加高斯模糊

    shift_x = random.randint(min_shift_size, max_shift_size)
    shift_y = random.randint(min_shift_size, max_shift_size)
    # 可能高斯分布为 （0, 6） 合适一点 +-3*sigma
    # shift_x = int(clamp_number(random.randn() * 6, min_shift_size, max_shift_size))
    # shift_y = int(clamp_number(random.randn() * 6, min_shift_size, max_shift_size))

    # shift_x = random.randint(min_shift_size, max_shift_size)
    # shift_y = random.randint(min_shift_size, max_shift_size)

    h, w = reflection.shape[0], reflection.shape[1]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    reflection_shifted = cv2.warpAffine(reflection, M, (w, h))
    attenuation = random.uniform(0.5, 1)
    # attenuation = random.uniform(0.5, 1)
    reflection = reflection * attenuation + reflection_shifted * (1 - attenuation)
    # shifted = max(abs(shift_x), abs(shift_y))
    # if shift_x >= 0 and shift_y >= 0:
    #     reflection = reflection[shifted:, shifted:]
    # elif shift_x >= 0 and shift_y < 0:
    #     reflection = reflection[:-shifted, shifted:]
    # elif shift_x < 0 and shift_y >= 0:
    #     reflection = reflection[shifted:, :-shifted]
    # else:
    #     reflection = reflection[:-shifted, :-shifted]
    reflection = reflection[shift_x:, shift_y:]
    reflection = cv2.resize(reflection, (int(w), int(h)))

    # rand_w = random.randint(0, abs(shift_y)) if shift_y != 0 else 0
    #
    # rand_h = random.randint(0, abs(shift_x)) if shift_x != 0 else 0
    # transmission = transmission[rand_w: w - abs(shift_y) + rand_w, rand_h: h - abs(shift_x) + rand_h]
    a = np.random.uniform(0.5, 1)
    transmission = a * transmission
    blended = transmission + reflection
    # blended = transmission + reflection
    if np.max(blended) > 1:
        m = blended[blended > 1]
        m = (np.mean(m) - 1) * 1.3
        reflection = np.clip(reflection - m, 0, 1)
        blended = np.clip(transmission + reflection, 0, 1)
    # cv2.imshow('blended', blended)
    # cv2.waitKey()
    #
    # cv2.imshow('t', transmission)
    # cv2.waitKey()
    #
    # cv2.imshow('r', reflection)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return transmission, reflection, blended

