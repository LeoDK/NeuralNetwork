# -*- coding:utf-8 -*-

from perceptron import *
from random import random
from abc import ABCMeta, abstractmethod
from sys import stdout
import pickle

def randList (length):
    ret = []
    for i in range(length):
        ret.append(random()-0.5)
    return ret


class Network (object):

    __metaclass__ = ABCMeta

    def __init__ (self, net, Perceptron_t, Output_t):
        """
        [net] defines the network layout : 
        it is a 2D array containing a series of layers, which contain neurons.
        This array may be initialised with 0s.
        Ex : net = [ [0, 0],
                     [0, 0, 0],
                     [0] ]
        -> network with 2 neurons in the first layer as inputs,
         3 neurons as perceptrons in the second,
         1 neuron as output in the third.
        """
        self.net = net 
        self.net[0] = [0.0] * len(self.net[0]) # Inputs
        self.examples = 0

        for i in range( 1, len(self.net)-1 ):
            for j in range( len(self.net[i]) ):
                self.net[i][j] = Perceptron_t( randList(len(net[i-1])), random()-0.5, self.net[i-1] )

        for i in range( len(self.net[-1]) ):
            self.net[-1][i] = Output_t( randList(len(net[-2])), random()-0.5, self.net[-2] )


    def __str__ (self):
        ret = ""
        for l in self.net:
            for elem in l:
                ret = ret + str(elem)
                ret = ret + "    "
            ret = ret + "\n"
        return ret


    def compute (self):
        """
        Calculate neurons's values.
        """
        for i in range( 1, len(self.net) ):
            for p in self.net[i]:
                p.compute(self.net[i-1])


    def process (self, input_):
        """
        Process an input and gives a prediction.
        """
        assert len(input_) == len(self.net[0])
        self.net[0] = input_
        self.compute()
        return self.net[-1]


    def backprop (self, y):
        """
        Basic backpropagation : minimizing cost.
        [y] : vector of what we want to have as an output.
        """
        self.net.append(y)
        for i in range( len(self.net)-2, 0, -1 ):
            for j in range( len(self.net[i]) ):
                self.net[i][j].simul( self.net[i-1], self.net[i+1], j, self.examples )
        del( self.net[-1] )


    def learn (self, input_, result):
        """
        Learn from one example.
        """
        assert len(input_) == len(self.net[0])
        self.examples += 1
        self.process(input_)
        self.backprop(result)


    def evolve (self):
        """
        Change actual network values after numerous simulations.
        """
        for l in self.net[1:]:
            for p in l:
                p.evolve()
        self.examples = 0
        self.compute()


    @abstractmethod
    def train (self, n_simul, n_cluster):
        pass

    @abstractmethod
    def test (self, n_tests):
        pass

    def save (self, name):
        with open("{}.pkl".format(name), "wb") as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load (name):
        with open("{}.pkl".format(name), "r") as save:
            net = pickle.load(save)
        return net


from mnist import MNIST
from PIL import Image
import numpy as np

class DigitRecognition (Network):

    def __init__ (self, net):
        Network.__init__(self, net, SigmoidPerceptron, SigmoidOutput)


    def train (self, n_simul, n_cluster):
        print("Starting...")
        mndata = MNIST("MNIST_db")
        images, labels = mndata.load_training()

        print("Simulations done : ")
        for i in range(n_simul):
            for j in range(n_cluster):
                rand = int( random() * 60000 )
                ans = [0]*9
                ans.insert( labels[rand],1 )
                self.learn( images[rand], ans )

            self.evolve()
            stdout.write( "\b" * len(str(i)) )
            stdout.write(str(i+1))
            stdout.flush()

        print("\n\nOK!")


    def test (self, n_tests):
        print("Starting...")
        mndata = MNIST("MNIST_db")
        images, labels = mndata.load_testing()
        good_results = 0

        for i in range(n_tests):
            print("************ Test {} ************".format(i)) 

            rand = int( random() * 10000 )
            ans = [0]*9
            ans.insert( labels[rand], 1 )
            out = self.process( images[rand] )

            print(mndata.display( images[rand] ))
            for o,a in zip(out,ans):
                print("{} : {}".format(o,a))

            out = map(lambda y:y.x, out)
            choice = out.index(max(out))
            print("Choice : {}".format(choice))
            print("\n")

            if choice == labels[rand]:
                good_results += 1

        print("OK!\n")
        print("####################################################")
        print("###############       Results    ###################")
        print("####################################################")
        print("{}/{} ({} %)".format( good_results, n_tests, float(good_results)/n_tests*100 ))


    def predictImage (self, path):
        """
        A bit buggy because you need to provide a digit with the exact same
        caracteristics as the ones used in MNIST dataset.
        """
        img = Image.open(path)
        matrix = np.array(img)
        matrix = matrix.tolist()

        input_ = []
        for line in matrix:
            for elem in line:
                input_.append( (elem[0] + elem[1] + elem[2])/3 )

        out = self.process(input_)
        out = map(lambda y:y.x, out)
        choice = out.index(max(out))

        for i in range(len(out)):
            print("{} : {}".format(i, out[i]))
        print("Choice : {}".format(choice))

        return out
