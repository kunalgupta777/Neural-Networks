## 1.) Using Neural Network Perform the implementation of AND, NAND, OR, and NOR

# Truth Table for AND Gate
and_gate = [(0,0,0),(0,1,0),(1,0,0),(1,1,1)]
# Truth Table for NAND Gate
nand_gate = [(0,0,1),(0,1,1),(1,0,1),(1,1,0)]
# Truth Table for OR Gate
or_gate = [(0,0,0),(0,1,1),(1,0,1),(1,1,1)]
# Truth Table for NOR Gate
nor_gate = [(0,0,1),(0,1,0),(1,0,0),(1,1,0)]

#defining the training function
def train(weights,gate):
    return [ weights[0]*gate[i][0]+weights[1]*gate[i][1] for i in range(len(gate))]

# calculating thresholds
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

weights_and =    [1,1]
weights_nand =   [1,1]
weights_or =     [1,1]
weights_nor =    [1,1]

#training the and_gate
vals_and = train(weights_and, and_gate)
vals_nand = train(weights_nand, nand_gate)
vals_or = train(weights_or, or_gate)
vals_nor = train(weights_nor, nor_gate)

values = [vals_and, vals_nand, vals_or, vals_nor]
gates_name = ["AND","NAND","OR","NOR"]
gates = [and_gate, nand_gate, or_gate, nor_gate]
for i in range(4):
    print "Values after computing outputs for " + gates_name[i] + " gate and the corresponding outputs:"
    print values[i]
    print [ gates[i][j][2] for j in range(4)]

    

#Calculating Thresholds for each gate
for i in range(4):
    print "Threshold for " + gates_name[i] + ":"
    print getthreshold(gates[i],values[i])



