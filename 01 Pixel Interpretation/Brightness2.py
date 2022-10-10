import cv2
import numpy as np
import time

# READ 1000 IMAGES
A = cv2.imread('grizzlypeakg.jpg',0)
m,n = A.shape
images = [A]*1000

# NAIVE METHOD
print("Start Naive Method")
start_time = time.time()
for i in range(1000):
    for j in range(m):
        for k in range(n):
            if images[i][j][k] <= 10:
                images[i][j][k] = 0
    if (i%25==0):
        print("%dth iteration"%i)
time_spent = time.time() - start_time
print('Naive Method Time : ', end= '')
print(time_spent)

# FASTER METHOD
print("Start Faster Method")
start_time2 = time.time()
for i in range(1000):
    images[i][images[i]<=10] = 0
time_spent2 = time.time() - start_time2
print('Faster Method Time : ', end='')
print(time_spent2)
print('speed-up factor: ', end= '')
print(time_spent/time_spent2, end= '')

