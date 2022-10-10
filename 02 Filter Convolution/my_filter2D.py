from time import gmtime
from unittest.mock import DEFAULT
from argon2 import hash_password
import cv2
from cv2 import BORDER_DEFAULT
import numpy as np


def my_filter2D(image, kernel, border_mode='constant'):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    ################
    # Your code here
    ################
    # Print error if image does not exist
    if image is None:
        print("Error: Error when opening image")
        print("Usage: my_filter2D.py")
        return -1
    # Print error if the kernel has an even dimension
    if (kernel.shape[0]%2==0 or kernel.shape[1]%2==0):
        print("Error: Even filters, outputs are undefined")
        print("Usage: my_filter2D.py")
        return -1
    # return original image if kernel is an identity matrix
    if (np.array_equal(kernel,np.eye(kernel.shape[0]))):
        return image
    pad_h = int((kernel.shape[0]-1)/2)
    pad_w = int((kernel.shape[1]-1)/2)
    image_padded = np.pad(image,pad_width=((pad_h,pad_h),(pad_w,pad_w),(0,0)),mode=border_mode)
    # if border_mode is set to 'reflect', the code pads the edges with the reflection
    if border_mode=='reflect':
        image_padded = np.pad(image,pad_width=((pad_h,pad_h),(pad_w,pad_w),(0,0)),mode='reflect')

    # convolution method
    convoloved_image = np.zeros_like(image)
    for k in range(3):
        for j in range(image_padded.shape[1]-kernel.shape[1]+1):
            for i in range(image_padded.shape[0]-kernel.shape[0]+1):
                total = np.sum(np.multiply(image_padded[i:i+kernel.shape[0],j:j+kernel.shape[1],k],kernel))
                convoloved_image[i,j,k] = total
    convoloved_image = np.abs(convoloved_image)
    # convoloved_image = fourier(image_padded,kernel,pad_h,pad_w)
    return convoloved_image

def fourier(image,kernel,pad_h,pad_w):
    # fft method
    fft_image_padded = np.fft.fft2(image,s=image.shape[0:2],axes=(0,1))
    fft_kernel = np.fft.fft2(kernel,s=image.shape[0:2],axes=(0,1))
    fft_kernel = np.expand_dims(fft_kernel, -1)
    fft_kernel = np.repeat(fft_kernel,3,axis=2)     # expand and duplicate kernel into 3d array

    fourier_multiplied = np.multiply(fft_image_padded,fft_kernel)
    calc = np.fft.ifft2(fourier_multiplied,axes=(0,1))
    return np.abs(calc)[2*pad_h:,2*pad_w:,:]


