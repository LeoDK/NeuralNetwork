# -*- coding:utf-8 -*-
"""
Train neural network to recognize handwritten digits with MNIST database.
Needs python-mnist v0.6 and pillow v5.2.0.
"""

from network import Network, DigitRecognition

#net = DigitRecognition( [[0]*784, [0]*16, [0]*16, [0]*10] )
net = Network.load("save/1")
net.test(500)
#net.train(500, 1)
#net.save("save")
