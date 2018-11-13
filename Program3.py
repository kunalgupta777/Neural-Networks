## 3.) Using Neural Network Perform any two class classification Problem
## Classify between 2 letters designed on a 3x3 matrix with classes 1 and -1
## Letters can be H and L

matrix_H = [1,-1,1,1,1,1,1,-1,1]
matrix_L =  [1,-1,-1,1,-1,-1,1,1,1]
target = [1,-1]
## Initialise the weights and bias as 0
weights_bias = [0] + [ 0 for i in range(9) ]

## Using Hebb's Rule to update weights
## for matrix_I
for i in range(10):
    if i==0:
        ## Update bias
        weights_bias[0]+=target[0]
    else:
        weights_bias[i]+=target[0]*matrix_H[i-1]
print "Bias and Weights after 1st training on I matrix:"
print weights_bias

## for matrix_U
for i in range(10):
    if i==0:
        ##Update bias
        weights_bias[0]+=target[1]
    else:
        weights_bias[i]+=target[1]*matrix_L[i-1]
print "Bias and Weights after 2nd training on U matrix:"
print weights_bias
        