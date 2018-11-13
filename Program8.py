## 8.) Implement Adaline Network
## An Adaline is a Neural Network that has a single Linear Activation Unit
## The network architecture is like McCulloch Pitts Model with the difference of the learning rule
## The Adaline employs the Widrow-Hoff rule for learning
'''
    w_i(new) = w_i(old) + alpha*(t - yin)x_i
    b_new = b_old + alpha*(t - yin)
    
    e = (t - yin)^2
'''
## The Network is trained until e is minimised
## The Adaline network uses bipolar inputs and outputs

import numpy as np

class Adaline:
    def __init__(self, input_neurons, alpha):
        self.input_neurons = input_neurons
        self.alpha = alpha
        
    def initialize(self):
        self.weights = np.array([0.1]*self.input_neurons)
        self.bias = 0.1
    
    def calc_yin(self, input_vector):
        yin = self.bias + sum(input_vector*self.weights)
        return yin
    def calc_weights(self, input_vector, target, yin):
        self.weights = self.weights + self.alpha*(target - yin)*input_vector
        self.bias = self.bias + self.alpha*(target - yin)
        
        return self.weights, self.bias
    def calc_error(self, target, yin):
        return (target - yin)**2
    
if __name__ == "__main__":
    
    net = Adaline(2,0.1)
    
    input_vectors = np.array( [ [-1,-1],[-1,1],[1,-1],[1,1] ])
    targets = np.array([-1, -1, -1, 1])
    
    print "------------Using Adaline Network to Model AND Gate----------------"
    
    epochs = int(raw_input("Enter the number of epochs:"))
    net.initialize()
    min_error = float("inf")
    w1, w2, b = 0, 0, 0   ## the final weights and biases that we'll find
    for _ in range(epochs):
        print "----------------------------------------------------------------"
        print "Epoch: ",str(_+1)
        
        print "x1\tx2\tt\tbias\t yin\t w1\t w2\t Error"
        
        avg_error = 0.00
        for i in range(len(input_vectors)):
            yin = net.calc_yin(input_vectors[i])
            weights, bias = net.calc_weights(input_vectors[i], targets[i], yin)
            error = net.calc_error(targets[i], yin)
            print str(input_vectors[i][0])+"\t"+str(input_vectors[i][1])+"\t"+str(targets[i])+"\t"+str(bias)+"\t"+str(yin)+"\t"+str(weights[0])+"\t"+str(weights[1])+"\t"+str(error)
            avg_error+=error
            if error < min_error:
                w1, w2, b  = weights[0], weights[1], bias
                min_error = error
        
        print "Average Error: {}".format(avg_error/4.00)
        print "---------------------------------------------------------------"
    
    print "Minimum Error", min_error
    print "Final Weights and Bias"
    print "w1 = {}, w2 = {}, bias = {}".format(w1, w2, b)
        
    
        