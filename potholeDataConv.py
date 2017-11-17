# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

#project dir
cdir = os.path.abspath(os.path.dirname(__file__))
print(cdir)

neg = '../pothole/train_data/Negative_data'
pos = '../pothole/train_data/Positive_data'
negt = '../pothole/Test_data/Negative'
post = '../pothole/Test_data/Positive'
neg_n = 0
hold = np.empty((1,19200))
negd = os.fsencode(neg)
posd = os.fsencode(pos)
negtd = os.fsencode(negt)
postd = os.fsencode(post)
negd_name = os.fsdecode(negd)
posd_name = os.fsdecode(posd)
negtd_name = os.fsdecode(negtd)
postd_name = os.fsdecode(postd)
print(negd_name)

for file in os.listdir(posd):
    filename = os.fsdecode(file)
    if filename.endswith(".JPG"):  
        img = mpimg.imread(posd_name + '/' + filename)
        
        img_small = Image.open(posd_name + '/' + filename).convert('LA')
        img_small.thumbnail((160, 120), Image.ANTIALIAS) # resizes image in-place
        plt.imshow(img_small)
        img_smalld = np.array(img_small.getdata())
        img_smalld = img_smalld.transpose()
        img_s = img_smalld.reshape(1,38400)
        img_s = img_s[:,:19200]
        #print(img_s)
        #imgplot2 = plt.imshow(img_small)
        #img1d = img.reshape(1,30470400)
        #print(img1d)
        hold = np.vstack((hold, img_s))
        #print(hold.shape)
        #print(f)
        neg_n += 1
        if(neg_n == 1000):
            break
        continue
    else:
        continue
#print(os.path.dirname(negd))
#print(hold.shape)
#imgplot = plt.imshow(img)
#delete first 'empty row'
hold = np.delete(hold, (0), axis=0)
holdd = hold[:500,:]
holdt = hold[500:1000,:]
print(hold.shape)
#hold = hold[:100,:]
#print(hold.shape)
np.save('pothole_data_pos_500_g', holdd)
np.save('pothole_data_post_500_g', holdt)
#print(img1d.shape)
'''
for file in os.listdir(posd):
    filename = os.fsdecode(file)
    if filename.endswith(".JPG"): 
        print(os.path.join(cdir, filename))
        neg_n += 1
        continue
    else:
        continue
'''