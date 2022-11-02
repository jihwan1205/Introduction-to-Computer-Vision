################################################################
# WARNING
# --------------------------------------------------------------
# When you submit your code, do NOT include blocking functions
# in this file, such as visualization functions (e.g., plt.show, cv2.imshow).
# You can use such visualization functions when you are working,
# but make sure it is commented or removed in your final submission.
#
# Before final submission, you can check your result by
# set "VISUALIZE = True" in "hw3_main.py" to check your results.
################################################################
from curses import window
from shutil import ReadError
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap
from regex import P
from sympy import denom, numer
from utils import normalize_points
import numpy as np
import cv2
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
from tqdm import tqdm

#=======================================================================================
# Your best hyperparameter findings here
WINDOW_SIZE = 7
DISPARITY_RANGE = 40
AGG_FILTER_SIZE = 100



#=======================================================================================
def bayer_to_rgb_bilinear(bayer_img):
    ################################################################

    h,w = bayer_img.shape
    mask = np.zeros((3,h,w))

    mask[0][0::2,0::2] = 1  # red
    mask[1][0::2,1::2] = 1  # green
    mask[1][1::2,0::2] = 1  # green
    mask[2][1::2,1::2] = 1  # blue

    kernel_g = np.array([[0,1/4,0],[1/4,1,1/4],[0,1/4,0]])
    kernel_rb = np.array([[1/4,1/2,1/4],[1/2,1,1/2],[1/4,1/2,1/4]])
    
    R = convolve(bayer_img * mask[0], kernel_rb,mode='constant',cval=0)
    G = convolve(bayer_img * mask[1], kernel_g,mode='constant',cval=0)
    B = convolve(bayer_img * mask[2], kernel_rb,mode='constant',cval=0)

    rgb_img = np.stack([R,G,B],axis=2).astype(np.uint8)

    ################################################################
    return rgb_img



#=======================================================================================
def bayer_to_rgb_bicubic(bayer_img):
    # Your code here
    ###############################################################

    h, w = bayer_img.shape

    maskR = np.zeros((h,w))
    maskG = np.zeros((h,w))
    maskB = np.zeros((h,w))
    maskR[0::2,0::2] = 1  # red
    maskG[0::2,1::2] = 1  # green
    maskG[1::2,0::2] = 1  # green
    maskB[1::2,1::2] = 1  # blue

    R = np.multiply(maskR, bayer_img)
    G = np.multiply(maskG, bayer_img)
    B = np.multiply(maskB, bayer_img)

    border = 'constant'
    Rpad = np.pad(R,((3,3),(3,3)),border)
    Gpad = np.pad(G,((3,3),(3,3)),border)
    Bpad = np.pad(B,((3,3),(3,3)),border)

    for i in tqdm(range(0,h,2)):
        for j in range(0,w,2):
            temp1 = np.array([[1,0,0,0],[0,0,1,0],[-3,3,-2,-1],[2,-2,1,1]])
            temp2 = np.array([[1,0,-3,2],[0,0,3,-2],[0,1,-2,1],[0,0,-1,1]])
            def p(a,x,y):
                xarray = np.array([1,x,x**2,x**3])
                yarray = np.array([1,y,y**2,y**3])
                return xarray @ a @ yarray
            def q(x,m,n,k,l):
                return m*(x**3) + n*(x**2) + k*x + l
            # Red
            f00 = Rpad[i+3][j+3]
            f01 = Rpad[i+3][j+5]
            f10 = Rpad[i+5][j+3]
            f11 = Rpad[i+5][j+5]
            fx00 = (Rpad[i+3][j+5] - Rpad[i+3][j+1])/2
            fx01 = (Rpad[i+5][j+5] - Rpad[i+5][j+1])/2
            fx10 = (Rpad[i+3][j+7] - Rpad[i+3][j+3])/2
            fx11 = (Rpad[i+5][j+7] - Rpad[i+5][j+3])/2
            fy00 = (Rpad[i+5][j+3] - Rpad[i+1][j+3])/2
            fy01 = (Rpad[i+7][j+3] - Rpad[i+3][j+3])/2
            fy10 = (Rpad[i+5][j+5] - Rpad[i+1][j+5])/2
            fy11 = (Rpad[i+7][j+5] - Rpad[i+3][j+5])/2
            fxy00 = (fx01-(Rpad[i+1][j+5]-Rpad[i+1][j+1])/2)/2
            fxy01 = ((Rpad[i+7][j+5]-Rpad[i+7][j+1])/2-fx00)/2
            fxy10 = (fx11-(Rpad[i+1][j+7]-Rpad[i+1][j+3])/2)/2
            fxy11 = ((Rpad[i+7][j+7]-Rpad[i+7][j+3])/2-fx10)/2
            Red_f = np.array([[f00,f01,fy00,fy01],[f10,f11,fy10,fy11],\
                [fx00,fx01,fxy00,fxy01],[fx10,fx11,fxy10,fxy11]])
            Red_a = temp1 @ Red_f @ temp2
            R[i][j] = Rpad[i+3,j+3]
            R[i+1][j] = p(Red_a,0,1/2)
            R[i][j+1] = p(Red_a,1/2,0)
            R[i+1][j+1] = p(Red_a,1/2,1/2)
           
            # Green
            G[i][j] = (1/16)*Gpad[i+3][j] + (9/16)*Gpad[i+3][j+2] + (7/16)*Gpad[i+3][j+4] + (-1/16)*Gpad[i+3][j+6]
            G[i+1][j] = Gpad[i+4][j+3]
            G[i][j+1] = Gpad[i+3][j+4]
            G[i+1][j+1] = (1/16)*Gpad[i+4][j+1] + (9/16)*Gpad[i+4][j+3] + (7/16)*Gpad[i+4][j+5] + (-1/16)*Gpad[i+4][j+7]
           
            # Blue
            f00 = Bpad[i+2][j+2]
            f01 = Bpad[i+2][j+4]
            f10 = Bpad[i+4][j+2]
            f11 = Bpad[i+4][j+4]
            fx00 = (Bpad[i+2][j+4] - Rpad[i+2][j])/2
            fx01 = (Bpad[i+4][j+4] - Bpad[i+4][j])/2
            fx10 = (Bpad[i+2][j+6] - Bpad[i+2][j+2])/2
            fx11 = (Bpad[i+4][j+6] - Bpad[i+4][j+2])/2
            fy00 = (Bpad[i+4][j+2] - Bpad[i][j+2])/2
            fy01 = (Bpad[i+6][j+2] - Bpad[i+2][j+2])/2
            fy10 = (Bpad[i+4][j+4] - Bpad[i][j+4])/2
            fy11 = (Bpad[i+6][j+4] - Bpad[i+2][j+4])/2
            fxy00 = (fx01-(Bpad[i][j+4]-Bpad[i][j])/2)/2
            fxy01 = ((Bpad[i+6][j+4]-Bpad[i+6][j])/2-fx00)/2
            fxy10 = (fx11-(Bpad[i][j+6]-Bpad[i][j+2])/2)/2
            fxy11 = ((Rpad[i+6][j+6]-Rpad[i+6][j+2])/2-fx10)/2
            Blue_f = np.array([[f00,f01,fy00,fy01],[f10,f11,fy10,fy11],\
                [fx00,fx01,fxy00,fxy01],[fx10,fx11,fxy10,fxy11]])
            Blue_a = temp1 @ Blue_f @ temp2
            B[i][j] = p(Blue_a,1/2,1/2)
            B[i+1][j] = p(Blue_a,1/2,1)
            B[i][j+1] = p(Blue_a,1,1/2)
            B[i+1][j+1] = Bpad[i+4][j+4]

    rgb_img = np.stack((R,G,B), axis=2)
    rgb_img = None
    ################################################################
    return rgb_img



#=======================================================================================
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]
    # Your code here
    ################################################
    
    pts1 = np.column_stack((pts1, [1] * pts1.shape[0]))
    pts2 = np.column_stack((pts2, [1] * pts2.shape[0]))
    pts1,T = normalize_points(pts1.T,2)
    pts2,Tprime = normalize_points(pts2.T,2)
    pts1 = pts1.T
    pts2 = pts2.T
    A = np.zeros((8,9))
    for i in range(pts1.shape[0]):
        x = np.reshape(np.array([pts1[i][0],pts1[i][1],1]),(1,3))
        xprime = np.reshape(np.array([pts2[i][0],pts2[i][1],1]),(3,1))
        temp = np.reshape(np.dot(xprime,x),(1,9))
        A[i,:] = temp
    AtA = np.dot(A.T,A)
    eigval, eigvec = np.linalg.eig(AtA)
    idx = np.argmin(eigval)
    f = eigvec[:,idx]
    F = np.reshape(f,(3,3))
    u,s,vt = np.linalg.svd(F)
    s[-1] = 0
    s = np.diag(s)
    fundamental_matrix_norm = np.dot(u,np.dot(s,vt))
    fundamental_matrix = np.dot(Tprime.T,np.dot(fundamental_matrix_norm,T))
    
    ################################################################
    return fundamental_matrix



#=======================================================================================
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # You should get un-cropped image.
    # In order to superpose two rectified images, you need to create certain amount of margin.
    # Which means you need to do some additional things to get fully warped image (not cropped).
    ################################################
    h,w = img1.shape[:2]
    corners = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
    corners1 = cv2.perspectiveTransform(np.float32([corners]),h1)[0]
    corners2 = cv2.perspectiveTransform(np.float32([corners]),h2)[0]

    bx1,by1,bwidth1,bheight1 = cv2.boundingRect(corners1)
    bx2,by2,bwidth2,bheight2 = cv2.boundingRect(corners2)

    translation = np.array([[1,0,-bx1],[0,1,-by1],[0,0,1]])
    h1 = np.dot(h1,translation)
    img1_rectified = cv2.warpPerspective(img1,h1,(bwidth2,bheight2))

    translation = np.array([[1,0,-bx2],[0,1,-by2],[0,0,1]])
    h2 = np.dot(h2,translation)
    img2_rectified = cv2.warpPerspective(img2,h2,(bwidth2,bheight2))
    
    ################################################
    return img1_rectified, img2_rectified




#=======================================================================================
def calculate_disparity_map(img1, img2):
    # First convert color image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # You have to get disparity (depth) of img1 (left)
    # i.e., I1(u) = I2(u + d(u)),
    # where u is pixel positions (x,y) in each images and d is dispairty map.
    # Your code here
    ################################################
    
    h,w = img1_gray.shape[:2]
    cost_volume = np.zeros((h,w,DISPARITY_RANGE))
    disparity_map = np.zeros((h,w))
    
    pad_size = int((WINDOW_SIZE-1)/2)
    img1_gray = np.pad(img1_gray,((pad_size,pad_size),(pad_size,pad_size)),'reflect')
    img2_gray = np.pad(img2_gray,((pad_size,pad_size),(pad_size,pad_size)),'reflect')
    for i in tqdm(range(h)):
        if(i%WINDOW_SIZE!=0):
            continue
        for j in range(w):
            if (j%WINDOW_SIZE!=0):
                continue
            window1 = img1_gray[i:i+WINDOW_SIZE,j:j+WINDOW_SIZE]  
            window1 = window1 - np.mean(window1)    # zero-mean NCC
            for k in range(DISPARITY_RANGE):
                if(j-k<0):
                    break
                window2 = img2_gray[i:i+WINDOW_SIZE,j-k:j-k+WINDOW_SIZE] 
                window2 = window2 - np.mean(window2) # zero-mean NCC
                ncc = 0
                if(np.sqrt(np.sum(window1*window1))*np.sqrt(np.sum(window2*window2))) != 0:
                    ncc = np.sum(window1*window2) / (np.sqrt(np.sum(window1*window1))*np.sqrt(np.sum(window2*window2)))
                cost_volume [i:i+WINDOW_SIZE, j-DISPARITY_RANGE:j+WINDOW_SIZE-DISPARITY_RANGE,k] = ncc

    # Cost Aggregation
    blur_size = (int((h-1)/3), int((h-1)/3))
    for i in tqdm(range(DISPARITY_RANGE)):
        cost_volume[:,:,i] = cv2.GaussianBlur(cost_volume[:,:,i],blur_size,0,0)
    for i in range(h):
        for j in range(w):
            disparity_map[i][j] = -np.argmax(cost_volume[i][j][:])


    ################################################################
    return disparity_map


#=======================================================================================
# Anything else:
