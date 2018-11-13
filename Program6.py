## 6.) Implement McCulloch-Pits Model of Neural Networks
## Quick Intro to McCulloch-Pitts Model of Neural Network
## The neural network has 'n' input neurons and one output neuron.
## Depending on the sign of the weight, the neuron can act as an excitatory neuron ( which activates the input stimulus )
## or inhibitory neuron, which deactivates the input.
## Let there be 'n' input neurons of which, 'm' are excitatory with weight 'w' (w>0) and 'n-m' inhibitory, with weight 'p' (p<0)
## The output of the first layer, assuming x = (x1, x2, x3, .... xn) is the input vector is

## yin = b + sum(i = 1 to m)xi*w + sum(i=m+1 to n)xi*p

## The final output is given by the activation of the net input to the output neuron,
## i.e. y = f(yin) = { 1  if yin >= threshold; 0 otherwise
## The McCulloch Model uses binary inputs and outputs
class McCullochs:
    
    def __init__(self, excitatory_neurons, inhibitory_neurons, w, p):
        self.excitatory_neurons = excitatory_neurons
        self.inhibitory_neurons = inhibitory_neurons
        self.w = w
        self.p = p
    
    def calc_yin(self, input_vector):
        s1 = self.w*sum(input_vector[:self.excitatory_neurons])
        s2 = self.p*sum(input_vector[self.excitatory_neurons:])
        yin = s1 + s2
        return yin
    
    def calc_y(self,yin, threshold):
        if yin> threshold:
            return 1
        else:
            return 0
        
if __name__ == "__main__":
    
    ## Now, we'll illustrate a problem that can be modelled using the McCulloch Pitt's Model
    ## Consider Modelling the AND Gate, with 2 input neurons and 1 output neuron
    ## The Truth Table is:
    ## x1 x2 y
    ## 0  0  0
    ## 0  1  0
    ## 1  0  0
    ## 1  1  1
    network = McCullochs(2,0,1,0)
    
    ## Now, we'll calculate the, the yin for each input pair
    input_pairs = [(0,0),(0,1),(1,0),(1,1)]
    outputs = [0,0,0,1]
    yins = []
    i = 0
    print "Modelling McCulloch Pitts Neuron"
    print "Input Pair\t Calculated yin\t Target Output"
    for input_pair in input_pairs:
        yins.append(network.calc_yin(input_pair))
        print str(input_pair)+"\t\t "+str(yins[-1])+"\t\t "+str(outputs[i])
        i+=1
        
    ## Now, we'll calculate the threshold for the above function
    gate = [(input_pairs[i][0],input_pairs[i][1],outputs[i]) for i in range(4)]
    threshold = getthreshold(gate, yins)
    print "Threshold for AND gate is:", threshold
    print"Calculated y = f(yin)"
    print "Input Pair\t Calculated yin\t Target Output \t Calculated y = f(yin)"
    for i in range(len(input_pairs)):
        print str(input_pairs[i])+"\t\t "+str(yins[i])+"\t\t "+str(outputs[i])+"\t\t "+str(network.calc_y(yins[i],threshold))
        
