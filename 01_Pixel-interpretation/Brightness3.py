import cv2
import numpy as np
import time

# Read Color Image
A = cv2.imread('grizzlypeak.jpg')
m,n,c = A.shape
new_A = np.mean(A,axis=2, keepdims=True)

# Naive Method
start_time = time.time()
for i in range(m):
   for j in range(n):
        if new_A[i][j]<=10:
            A[i][j][0] = 0
            A[i][j][1] = 0
            A[i][j][2] = 0
time_spent = time.time() - start_time
print('original time : ', end= '')
print(time_spent)

# Logical Indexing Method
start_time2 = time.time()
B = new_A<=10
A = np.concatenate((B,B,B),axis=2)
time_spent2 = time.time() - start_time2
print('faster time : ', end='')
print(time_spent2)
print('speed-up factor: ', end= '')
print(time_spent/time_spent2, end= '')

