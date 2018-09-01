# -*- coding:utf-8 -*-

from math import e
from abc import ABCMeta


class Perceptron (object):

    """
    Perceptron object.

     a1 o 
         \ 
          \ w1
         w2\ 
     a2 o---O (b) ----> x
           /
          / w3
         /
     a3 o

     |
     |

    prev_l
    """

    def __init__ (self, w, b, f, d_f, prev_l):
        """
        [w] : input weights vector.
        [b] : bias.
        [f] : activation function.
        [d_f] : activation function derivative.
        [prev_l] : previous layer of neuron, as a vector.
        """
        self.w = w
        self.b = b
        self.f = f
        self.d_f = d_f

        self.compute(prev_l)
        self.clear()


    def __mul__ (self, n):
        return self.x * n

    def __str__ (self):
        return str(self.x)

    def compute (self, prev_l):
        """
        Compute perceptron value.
        [prev_l] : previous layer of neuron, as a vector.
        """
        #        n
        # x = f( S(ai wi) + b )
        #       i=0
        self.x = 0.0
        for i in range( len(prev_l) ):
            self.x += prev_l[i] * self.w[i]
        self.x += self.b
        self.x = self.f( self.x )


    def simul (self, prev_l, next_l, index, trainings):
        """
        After a simulation, calculates how to change parameters to minimize cost (or error),
        and improve network's skills.
        [prev_l] : previous layer of neuron, as a vector.
        [next_l] : next layer of neuron, as a vector.
        [index] : index of neuron in layer.
        [trainings] : number of trainings realized.
        """
        #     n
        # z = S(ai wi) + b
        #    i=0
        #
        # So x = f(z)
        z = 0.0
        for i in range( len(prev_l) ):
            z = prev_l[i] * self.w[i]
        z += self.b

        # dc / dx
        for i in range( len(next_l) ):
            self.d_x_save += next_l[i].w[index] * self.d_f(z) * next_l[i].d_x
        self.d_x = self.d_x_save / trainings

        # dc / dw
        for i in range(len(self.w)):
            self.d_w_save[i] += prev_l[i] * self.d_f(z) * self.d_x
            self.d_w[i] = self.d_w_save[i] / trainings

        # dc / db
        self.d_b_save += self.d_f(z) * self.d_x
        self.d_b = self.d_b_save / trainings


    def evolve (self):
        """
        Apply average changes to minimize error, calculated during [cost] function call(s).
        We add the negative gradient or derivative to the weights and to the bias.
        """
        for i in range(len(self.w)):
            self.w[i] -= self.d_w[i]
        self.b -= self.d_b
        self.clear()


    def clear (self):
        """
        Clean the results of past simulations.
        """
        self.d_w = [0.0] * len(self.w)
        self.d_w_save = [0.0] * len(self.w)
        self.d_b = 0.0
        self.d_b_save = 0.0
        self.d_x = 0.0
        self.d_x_save = 0.0



def sigmoid (x):
    if x > 100:  # Prevent OverflowError
        return 1
    if x < -100:
        return 0
    return 1/( 1+e**(-x) )

def d_sigmoid (x):
    return e**(-x) / ( 1+e**(-x) )**2


class SigmoidPerceptron (Perceptron):

    def __init__ (self, w, b, prev_l):
        Perceptron.__init__(self, w, b, sigmoid, d_sigmoid, prev_l)


def ReLU (x):
    return max(0,x)

def d_ReLU (x):
    if x > 0:
        return 1
    if x < 0:
        return 0
    return ValueError("ReLU derivative is undefined at x=0")


class ReLUPerceptron (Perceptron):

    def __init__ (self, w, b, prev_l):
        Perceptron.__init__(self, w, b, ReLU, d_ReLU, prev_l)


class Output (Perceptron):

    def simul (self, prev_l, y, index, trainings):
        """
        [prev_l] : previous layer of neuron, as a vector.
        [y] is the desired output, as a vector (considered as last and next layer).
        """
        # Value derivative : dc / dx = d (x - y)^2 / dx = 2*(x-y) in the last layer
        self.d_x_save += 2*(self.x - y[index])
        self.d_x = self.d_x_save / trainings
        self.__class__.__bases__[1].simul(self, prev_l, [], index, trainings)


class SigmoidOutput (Output, SigmoidPerceptron):
    pass

class ReLUOutput (Output, ReLUPerceptron):
    pass
