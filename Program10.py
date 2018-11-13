## 10.) Implement Auto Associative Neural Networks
## Auto associative neural nets are a special type of nets which store pattern associations
## Here, the output vector is identical to the input vector
## Both, the input layer and the output layer have 'n' neurons
## Hebbian Learning Rule can be applied to find the weights
print "Auto Associative Networks"
n = int(raw_input("Enter n:"))
X = [ random.choice([-1,1]) for i in range(n)]
Y = [ random.choice([-1,1]) for i in range(n)]
print "Input Vector is",X
print "Output Vector is",Y
weights = [ [ 0 for _ in range(n)] for _ in range(n)]
## Training Phase
for i in range(n):
    for j in range(n):
        weights[i][j]+=X[i]*Y[j]
print "Weights after Training:"
print weights
## Testing Phase
test = [ random.choice([-1,1]) for i in range(n)]
print "Test Input",test
def f(yinj):
    if yinj > 0:
        return 1
    else:
        return -1
outs= []
for j in range(n):
    yinj = 0
    for i in range(n):
        yinj+=test[i]*weights[i][j]
    yin = f(yinj)
    outs.append(yin)
print "Testing Output",outs
        
    