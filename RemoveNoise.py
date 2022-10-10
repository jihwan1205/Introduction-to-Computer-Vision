import cv2
import numpy as np
import time

A = cv2.imread('grizzlypeakg.jpg',0)
m1, n1 = A.shape
start_time = time.time()
for i in range(m1):
   for j in range(n1):
      if A[i,j] <= 10:
        A[i,j] = 0
time_spent = time.time() - start_time
print('original time : ', end= '')
print(time_spent)

A = cv2.imread('grizzlypeakg.jpg',0)
m1, n1 = A.shape
start_time = time.time()
A[A<=10] = 0
time_spent2 = time.time() - start_time
print('faster time : ', end='')
print(time_spent2)
print('speed-up factor: ', end= '')
print(time_spent/time_spent2, end= '')

