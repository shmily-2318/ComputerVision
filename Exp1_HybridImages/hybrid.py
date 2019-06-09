import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')
import numpy as np
import copy
import math
import cv2


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x node dimensions, with both m and node being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x node), with m and node both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    kernel_h, kernel_w = kernel.shape
    if img.ndim == 3:
        img_h, img_w, img_c = img.shape

        eimg = np.zeros((img_h + kernel_h - 1, img_w + kernel_w - 1, img_c))
        eimg_h, eimg_w, eimg_c = eimg.shape
        eimg[kernel_h // 2:eimg_h - (kernel_h // 2), kernel_w // 2:eimg_w - (kernel_w // 2), :] = img

        nimg = copy.deepcopy(eimg)

        for i in range(kernel_h // 2, eimg_h - (kernel_h // 2)):
            for j in range(kernel_w // 2, eimg_w - (kernel_w // 2)):
                sum = np.zeros((1, 3))
                """
                for m in range(0, kernel_h):
                    for node in range(0, kernel_w):
                        sum += eimg[i - kernel_h // 2 + m, j - kernel_w // 2 + node, :] * kernel[m, node]
                """
                for k in range(0, 3):
                    sum[0, k] = np.sum(eimg[i - kernel_h // 2:i + kernel_h // 2 + 1,
                                            j - kernel_w // 2:j + kernel_w // 2 + 1, k] * kernel)
                nimg[i, j, :] = sum

        nimg = nimg[kernel_h // 2:eimg_h - (kernel_h // 2), kernel_w // 2:eimg_w - (kernel_w // 2), :]
    else:
        img_h, img_w = img.shape

        eimg = np.zeros((img_h + kernel_h - 1, img_w + kernel_w - 1))
        eimg_h, eimg_w = eimg.shape
        eimg[kernel_h // 2:eimg_h - (kernel_h // 2), kernel_w // 2:eimg_w - (kernel_w // 2)] = img

        nimg = copy.deepcopy(eimg)

        for i in range(kernel_h // 2, eimg_h - (kernel_h // 2)):
            for j in range(kernel_w // 2, eimg_w - (kernel_w // 2)):
                sum = 0
                """
                for m in range(0, kernel_h):
                    for node in range(0, kernel_w):
                        sum += eimg[i - kernel_h // 2 + m, j - kernel_w // 2 + node] * kernel[m, node]
                """
                sum = np.sum(eimg[i - kernel_h // 2:i + kernel_h // 2 + 1,
                                j - kernel_w // 2:j + kernel_w // 2 + 1] * kernel)
                nimg[i, j] = sum

        nimg = nimg[kernel_h // 2:eimg_h - (kernel_h // 2), kernel_w // 2:eimg_w - (kernel_w // 2)]

    return nimg


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x node), with m and node both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    kernel = np.rot90(kernel, 2)
    return cross_correlation_2d(img, kernel)


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    gauss_kernel = np.zeros((height, width))
    center_kernel_x = height // 2
    center_kernel_y = width // 2

    s = 2 * (sigma ** 2)
    sum_vals = 0

    for i in range(0, height):
        for j in range(0, width):
            x = i - center_kernel_x
            y = j - center_kernel_y
            gauss_kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s) / (s * math.pi)
            sum_vals += gauss_kernel[i, j]
    gauss_kernel /= sum_vals

    return gauss_kernel


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    gauss_kernel = gaussian_blur_kernel_2d(sigma, size, size)
    low_pass_img = convolve_2d(img, gauss_kernel)

    #cv2.imshow('low_pass', (low_pass_img * 255).clip(0, 255).astype(np.uint8))

    return low_pass_img


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    gauss_kernel = gaussian_blur_kernel_2d(sigma, size, size)
    high_pass_img = img - convolve_2d(img, gauss_kernel)

    #cv2.imshow('high_pass', (high_pass_img * 255).clip(0, 255).astype(np.uint8))

    return high_pass_img


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


if __name__ == '__main__':
    img1 = cv2.imread('cat.jpg', -1)
    img2 = cv2.imread('dog.jpg', -1)

    hybrid = create_hybrid_image(img1, img2, 11, 13, 'high', 11, 13, 'low', 0.5)

    cv2.imshow('Image', hybrid)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite('hybrid.jpg', hybrid)
