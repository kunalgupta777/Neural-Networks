## 7.) Implement Hebb Network

## A Hebb network is a neural network that employs Hebb Learning Rule, which is:
'''
    w_i(new) = w_i(old) + x_i*y
    bias(new) = bias(old)+ y
'''
## Modelling the AND gate again, but this time, we will use Hebbian Lerning rule to finalise weights instead of randomly guessing 
## them
import matplotlib.pyplot as plt
import numpy as np
and_gate_inputs = [ (1,1),(1,-1),(-1,1),(-1,-1) ]
and_gate_outputs = [ 1, -1, -1, -1 ]

w1 = w2 = b = 0

print "Applying Hebbian Learning Rule to model AND gate"
print "Inputs\t   \t Output \t Weights and bias(initialised to 0)"
print "x1\t x2\t y \t \t w1\tw2\tb"

weights_bias = []
for i in range(4):
    x1 = and_gate_inputs[i][0]
    x2 = and_gate_inputs[i][1]
    y = and_gate_outputs[i]
    w1 = w1 + x1*y
    w2 = w2 + x2*y
    b = b + y
    weights_bias.append((w1, w2, b))
    print "{}\t {}\t {} \t \t {}\t{}\t{}".format(x1,x2,y,w1,w2,b)
    
## Now, showing plots of each weight value using linear separbility concept
print "Line Plot showing linear separability of AND gate"
for (w1, w2, b) in weights_bias:
    print "w1 = {}, w2 = {}, b = {}".format(w1, w2, b)
    ## plotting the 4 input pairs
    x = [1,1,-1,-1]
    y = [1,-1,1,-1]
    plt.plot(x,y,'mo')
    for a,b in zip(x,y):
        plt.text(a,b,"( {}, {} )".format(a,b))
    X = np.arange(-1,2)
    m = -w1/w2
    c = -b/w2
    print m,c
    Y = m*X + c
    print X,Y
    plt.plot(X,Y)
    plt.xlabel('x2')
    plt.ylabel('x1')
    plt.show()
    
        
    
    
    


