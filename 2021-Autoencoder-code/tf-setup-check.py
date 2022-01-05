# Requirements: Ubuntu 16.04
# Please execute the following commands in a shell to install all the needed dependencies after that, execute this file:
# python3 tf-setup-check.py

'''
# Use apt to install and then remove numpy and matplotlib so we can have all the dependencies,
# then we install them using pip3 for the latest version

sudo apt install python3 python3-pip python3-numpy python3-matplotlib
sudo apt remove python3-numpy python3-matplotlib

sudo pip3 install --upgrade pip
sudo pip3 install numpy matplotlib
sudo pip3 install tensorflow

optional:
sudo pip3 install jupyter

# Install your favorite code editor!
'''

# For other OS please install the same packages.
# Have a look at Conda https://conda.io/docs/user-guide/install/windows.html
# and follow the following guide: https://www.tensorflow.org/install/

import tensorflow
print("tensorflow", tensorflow.__version__)

print("keras", tensorflow.keras.__version__)

from tensorflow.keras.datasets import mnist
print("caching data from mnist dataset")
(x_train, _), (x_test, _) = mnist.load_data()

print()
print("All OK")
