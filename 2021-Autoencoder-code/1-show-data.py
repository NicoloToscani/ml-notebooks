#!/usr/bin/env python
# coding: utf-8

# # MNIST Database
# 
# MNIST is a simple computer vision dataset that consists of images of handwritten digits and also includes labels for each image, telling us which digit it is.
# 
# Images are 28x28 pixel grayscale.
# 
# The MNIST database contains 60,000 training images and 10,000 testing images and it is hosted by Yann LeCun here: http://yann.lecun.com/exdb/mnist/
# 
# Best classification performance:
# * type: committee of 35 convolutional neural networks
# * test error rate	0.23 %
# * by: Ciresan et al. CVPR 2012
# 
# *Can you beat it?*
# *(we will not try that today)*
# 

# In[1]:


import tensorflow
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()


# In[2]:


print(x_train.shape)
print(x_test.shape)


# In[3]:


print( x_train[0])


# In[4]:


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# In[5]:


print( x_train[0])


# In[6]:


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


# In[8]:


import matplotlib.pyplot as plt

n = 20  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# In[ ]:




