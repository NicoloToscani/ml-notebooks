#!/usr/bin/env python
# coding: utf-8

# # Deep Autoencoder
# 
# 784 -> 128 -> 64 > 32 > 64 > 128 -> 784
# 
# ![](img/autoencoder_schema.jpg)
# 
# `Warning: since the dataset is quite easy there will not be major loss differences`

# In[1]:


import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt


# Let's create the Model

# In[2]:


# INPUT
input_img = Input(shape=(784,))

# ENCODER
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# DECODER
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# MODEL
autoencoder = Model(input_img, decoded)


# Quite easy, right? (If you want to do the same thing using TensorFlow directly it would take much more code)
# 
# Now we have to compile the Model

# In[3]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# As usual we need data, rescaled between 0 and 1, and reshaped in a 1D array

# In[4]:


from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# Now we have to train using the `fit` method

# In[5]:


autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# To use the model we can use `predict` on the `autoencoder` fitted Model

# In[6]:


decoded_imgs = autoencoder.predict(x_test)


# And now plot the results with the same code

# In[7]:


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




