## 4) Using Neural Network Implement Perceptron Learning Algorithm

## Using Perceptron Learning Algorithm for classifying 2 letters in a 3x3 matrix
matrix_I = [1,1,1,-1,1,-1,1,1,1]
matrix_F = [1,1,1,1,1,1,1,-1,-1]

# Initialising weights and biases
weights_bias = [0 for i in range(10)] 

target = [1,-1]

alpha = 0.1   ## learning rate

epochs = int(raw_input("Enter the number of epochs to train:"))

def f(yin):
    if yin>0:
        return 1
    elif yin==0:
        return 0
    else:
        return -1


for ep in range(epochs):
    print "Epoch #"+str(ep+1)+"beginning:"
    yinI = 0
    yinF = 0
    yinI = weights_bias[0]
    
    ## Training on 1st Matrix
    # Calculating yin
    
    for i in range(1,10):
        yinI+=matrix_I[i-1]*weights_bias[i]
    print "yin for I matrix:  ", yin
    
    # Target value 
    y = f(yinI)
    
    if y!= target[0]:
        ## weight updation starts
        weights_bias[0]+=alpha*target[0]
        for i in range(1,10):
            weights_bias[i]+=alpha*target[0]*matrix_I[i-1]
        print "New Calculated Bias and Weights:"
        print weights_bias
    else:
        print "No updation required"
        
    ## Training on 2nd matrix
    # Calculating yin
    yinF = weights_bias[0]
    for i in range(1,10):
        yinF+=matrix_F[i-1]*weights_bias[i]
    print "yin for F matrix: ", yin
    
    ## Target value
    y = f(yinF)
    if y!=target[1]:
        ## weight updation starts
        weights_bias[0]+=alpha*target[0]
        for i in range(1,10):
            weights_bias[i]+=alpha*target[0]*matrix_F[i-1]
        print "New Calculated Bias and Weights:"
        print weights_bias
    else:
        print "No Updation Required"
        
    if f(yinI)==target[0] and f(yinF)==target[1]:
        print "Network Converged Successfully!"

    
    print "Epoch #"+str(ep+1)+" ends:"
    

print "Final Bias and Weights:"
print weights_bias
    
    



