from keras.datasets import mnist
import numpy as np
from network import Network
from hyperdata import Data
import json

def one_hot(n):
    encoder = np.zeros(10)
    encoder[n] = 1
    return encoder

network = Network([784, 450, 250, 10], 
    Data(
        epochs=5, 
        learn_rate=0.01, 
        batch_size=32))
                         
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = [np.concatenate(x) / 255 for x in train_x]
test_x  = [np.concatenate(x) / 255 for x in test_x] 
train_y = [one_hot(y) for y in train_y]
test_y  = [one_hot(y) for y in test_y]

network.train(train_x, train_y)

SAMPLES = 10

for i in range(SAMPLES):
    num, out = network.predict(train_x[i])
    print(f'predicted: {num}, supposed to be {train_y[i]}!')
    print(f'distribution: {out}')
