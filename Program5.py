## 5) Using Neural Network Perform the implementation of XOR and XNOR

## For XOR Gate 

## The Function: f1 = x1(~x2)
f1 = [ (0,0,0), (0,1,0), (1,0,1), (1,1,0) ]
## The Function: f2 = (~x1)x2
f2 = [ (0,0,0), (0,1,1), (1,0,0), (1,1,0) ]
## The Function: f3 = f1+f2
f3 = [ (0,0,0) ,(0,1,1), (1,0,1), (0,0,0)]


## For XNOR Gate
## Function: f4 = x1x2
f4 = [ (0,0,0), (0,1,0), (1,0,0), (1,1,1) ]
## Function: f5 = (~x1)(~x2)
f5 = [ (0,0,1), (0,1,0), (1,0,0), (1,1,0) ]
## Function: f6 = f4 + f5
f6 = [ (0,1,1), (0, 0, 0), (0, 0, 0), (1, 0, 1)]


def train(weights, gate):
    return [ weights[0]*gate[i][0] + weights[1]*gate[i][1] for i in range(len(gate)) ]

## We model the XOR function as a 2 layer neural network
## The first layer is computes function f1, the second layer computes function f2, finally, the output is defined by ORing
## the outputs of the last layer.

def getthreshold(gate, value):
    l1 = []
    l2 = []
    for i in range(len(gate)):
        if gate[i][2]==0:
            l1.append(value[i])
        else:
            l2.append(value[i])
    
    if abs(max(l1)-min(l2))==1:
        return max(l1)
    else:
        return min(l1)
    
weights_f1 = [1,-1]
weights_f2 = [-1,1]
weights_f3 = [1, 1]
weights_f4 = [1, 1]
weights_f5 = [-1,-1]
weights_f6 = [1, 1]


vals_f1 = train(weights_f1, f1)
vals_f2 = train(weights_f2, f2)
vals_f3 = train(weights_f3, f3)
vals_f4 = train(weights_f4, f4)
vals_f5 = train(weights_f5, f5)
vals_f6 = train(weights_f6, f6)

print "----------------Modelling XOR Gate-------------------"
print "Values for f1:", vals_f1
print "Values for f2:", vals_f2
print "Values for f3:", vals_f3

print "Threshold for first layer:", getthreshold(f1, vals_f1)
print "Threshold for second layer:", getthreshold(f2, vals_f2)
print "Threshold for output:", getthreshold(f3, vals_f3)

print "----------------Modelling XNOR Gate--------------------"
print "Values for f4:", vals_f4
print "Values for f5:", vals_f5
print "Values for f6:", vals_f6

print "Threshold for first layer:", getthreshold(f4, vals_f4)
print "Threshold for second layer:", getthreshold(f5, vals_f5)
print "Threshold for output", getthreshold(f6, vals_f6)