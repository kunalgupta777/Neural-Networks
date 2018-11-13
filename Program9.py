## 9.) Implement Backpropagation in Neural Networks
## Backpropagation is a novel technique of upadating the weights and biases in a multilayer feed forward network, 
## where the activation function is differentiable
'''
    Architecture of Backpropagation Network (BPN)
    It has 2 layers:
    1.) Input Layer denoted by X - has 'n' neurons, which take input from the application - hence Xi means, ith input neuron
    2.) Hidden Layer denoted by Z - has 'p' neurons, which take input from the input layer - hence Zj means, the jth hidden neuron
    3.) Output Layer denotes by Y - has 'm' neurons, which take input from the hidden layer - hence Yk means, the kth output neuron
    
    The BPN architecture is a fully connected feedforward architecture, meaning, all neurons from the previous layer
    are connected to all the neurons of the next layer
    
    Some more terminology and symbology:-
    The hidden layer and the Output layer, besides the neurons, have a bias neuron as well, whose input is always one
    The weights from Input Layer to Hidden Layer are symbolised by 'v'
    and 
        vij - means the weight of the connection from ith input neuron to jth hidden neuron
    The weights from the Hidden Layer to the Output Layer are symbolised by 'w'
    and
        wjk - means the weight of the connection from the jth hidden neuron to the kth output neuron
    
    The functioning of the BPN is summarised in 3 Phases:
        1.) Feed Forward Phase
        2.) Back Propagation of the Error 
        3.) Weight and Bias Updation
    
    1.) Feed Forward Phase
    The net input for jth neuron in the hidden layer is given by:
        Zinj = v0j + sum(i =1 to n)Xi*Vij
    and the output the jth neuron will give would be the activation of the input, or
        Zj = f(Zinj)
    where f(x) = 1/( 1 + exp(-x)), the sigmoidal function
    
    Similarly, the net input for kth neuron in the output layer is given by:
        Yink = w0k + sum(j = 1 to p)Zj*wjk
    and the output of the kth neuron is likewise, 
        Yk = f(Yink)
    
    2.) The Backpropagation Phase
    
    Each output neuron has a corresponding target value tk and the output value the neuron gives Yk
    Obviously, we would want Yk to be as close as possible to tk, but making sure that the network doesn't overfits
    
    We define del-k to be the error associated with kth output neuron
        del-k = (tk - Yk)f'(Yink)
    We know that 
        f(x) = 1/( 1 + exp(-x))
        f'(x) = [-1/( 1 + exp(-x))^2][-exp(-x)] = exp(-x)/[1+exp(-x)]^2 = f(x)[exp(x)/[1+exp(-x)]] = f(x)[1-f(x)]
        hence, 
        f'(x) = f(x)[1-f(x)]
    As f(Yink) = Yk
    Or, 
    
        del-k = (tk - Yk)Yk(1 - Yk)
    Based upon this error, the change in the weight connecting Hidden and Output Layers , i.e. del-wjk is given by:
    
        del-wjk = alpha*del-k*Zj
    Similary, the change in bias is
        del-w0k = alpha*del-k
    
    Now, we will calculate the error associated with each Hidden Layer Neuron
    We, define, del-in-j as the backward input error to jth Hidden Neuron
        del-in-j = sum(k=1 to m)del-k*wjk
    As we did in feed forward phase, the output error propagated by jth Hidden Neuron, del-j is then:
        del-j = del-in-j*f'(Zinj)
    We know that f(Zinj) = Zj
    So,
        del-j = del-in-j*Zj[1 - Zj]
    Finally, we calculate the changes associated with the weights connecting Input and Hidden Layer
        del-vij = alpha*del-j*Xi
    and
        del-v0j = alpha*del-j
    
    3.) Weights and Bias Updation 
        We simply update the weights and bias
        vij = vij + del-vij
        v0j = v0j + del-v0j
        and
        wjk = wjk + del-wjk
        w0k = w0k + dek-w0k
    
    The running of all the 3 phases once, makes an epoch.
    In practice, we train the network till some predetermined epochs or till tk!=yk
'''
import numpy as np
import random
import math
## Let's Begin Now
class BPN:
    V = []
    W = []
    V0 = []
    W0 = []
    Zin = []
    Z = []
    Yin = []
    Y = []
    DelK = []
    DelW = []
    DelW0 = []
    DelinJ = []
    DelV = []
    DelV0 = []
    DelJ = []
    alpha = random.random()  # the learning rate
    def __init__(self, n, p, m):
        self.n = n
        self.p = p
        self.m = m
        # initialising the weights and biases
        self.V = self.DelV = np.array([ [ random.random() for _ in range(self.p)] for _ in range(self.n) ])
        self.W = self.DelW = np.array([ [ random.random() for _ in range(self.m)] for _ in range(self.p) ])
        self.V0 = self.DelV0 = np.array([ random.random() for _ in range(self.p) ])
        self.W0 = self.DelW0 = np.array([ random.random() for _ in range(self.m) ])
    
    def sigmoid(self,x):
        if type(x) is np.ndarray:
            return 1.0/(1.0 + np.exp(-x))
        else:
            return 1.0/(1.0 + math.exp(-x))
    
    
    def f(self, x):
        return self.sigmoid(x)
    
    def f_dash(self, x):
        return self.f(x)*(1 - self.f(x) )
    
    def feed_forward(self, X):
        print "-----------------------------Feed Forward Phase-----------------------------------"
        ## This is the feed forward function, refer to the writeup above for more information
        print "Weights Connecting Input and Hidden Layers"
        print self.V
        print "Biases"
        print self.V0
        print "Weights Connecting Hidden and Output Layers"
        print self.W
        print "Biases"
        print self.W0

        for j in range(self.p):
            zinj = self.V0[j] 
            for i in range(self.n):
                zinj+=X[i]*self.V[i][j]
            (self.Zin).append(zinj)
        ## Now, we'll calculate Zj = f(Zinj)
        print "Inputs to the Hidden Layer"
        print self.Zin
        self.Z = self.f(np.array(self.Zin))
        print "Outputs of the Hidden Layer"
        print self.Z
        for k in range(self.m):
            yink = self.W0[k]
            for j in range(self.p):
                yink+=self.Z[j]*self.W[j][k]
            (self.Yin).append(yink)
        print "Net Input to the Output Layer"
        print self.Yin
        self.Y = self.f(np.array(self.Yin))
        print "Net Output of the Output Layer"
        print self.Y
        print "------------------------------End Feed Forward Phase------------------------------"
        
    def backpropagate(self, targets, X):
        ## Now, we'll begin with the backpropagation phase
        print "-------------------------------Back Propagation Phase------------------------------"
        ## Errors with each output layer neuron
        for k in range(self.m):
            delk = (targets[k] - self.Y[k])*self.f_dash(self.Yin[k])
            self.DelK.append(delk)
        print "Output Errors for each Output Neuron"
        print self.DelK
        # Now we'll calculate the errors associated with the weights in Hidden and Output Layer
        for j in range(self.p):
            for k in range(self.m):
                self.DelW[j][k]  = self.alpha*self.DelK[k]*self.Z[j]
        # For Biases
        for k in range(self.m):
            self.DelW0[k] = self.alpha*self.DelK[k]
        
        print "Changes in weights for Hidden and Output Layer"
        print self.DelW
        print "Change in Biases"
        print self.DelW0
        ## Now,we'll calculate the backward error for each Hidden Layer Neuron
        for j in range(self.p):
            delj = sum( [ self.DelK[k]*self.W[j][k] for k in range(self.m) ] )
            self.DelinJ.append(delj)
        
        print "Input Errors with Each Hidden Layer Neuron"
        print self.DelinJ
        
        ## Output Errors for each Hidden Layer Neuron
        
        self.DelJ = np.array(self.DelinJ)*self.f_dash(np.array(self.Zin))
        
        print "Output Errors for Each Hidden Layer Neuron"
        print self.DelJ

        
        ## Finally, we'll calculate the errors associated with weights joining Input and Hidden Layer
        for i in range(self.n):
            for j in range(self.p):
                self.DelV[i][j] = self.alpha*self.DelJ[j]*X[i]
        ## For Biases
        for j in range(self.p):
            self.DelV0[j] = self.alpha*self.DelJ[j]
        print "Changes in Weights for Input and Hidden Layer"
        print self.DelV
        print "Change in Biases"
        print self.DelV0
        print "-------------------End of Back Propagation Phase-------------------------"
                
    def update(self):
        print "-------------------Updation Phase Starts----------------------------------"
        self.W  = self.W + self.DelW
        self.V  = self.V + self.DelV
        self.W0 = self.W0 + self.DelW0
        self.V0 = self.V0 + self.DelV0
        print "Updated Weights Connecting Input and Hidden Layers"
        print self.V
        print "Updated Biases"
        print self.V0
        print "Updated Weights Connecting Hidden and Output Layers"
        print self.W
        print "Updated Biases"
        print self.W0
        print "----------------------End of Updation Phase--------------------------------"
            
        
        
if __name__ == "__main__":
    print "Illustrating Backpropagation Network (BPN)"
    print "All Hyperparameters are randomly taken between 0 and 1 inclusive"
    n,p,m = map(int, raw_input("Enter number of neurons in Input, Hidden and Output Layers respectively (n,p,m):").split())
    epochs = int(raw_input("Enter the number of Epochs, you want to run the network for:"))
    
    bpn = BPN(n,p,m)
    X = [ random.random() ]*n           # the input vector
    targets = [ random.random() ]*m     # the output vector
    for i in range(epochs):
        print "------------------------------EPOCH {}--------------------------------".format(i+1)
        bpn.feed_forward(X)
        bpn.backpropagate(targets, X)
        bpn.update()
        print "------------------------------END--------------------------------------"

        