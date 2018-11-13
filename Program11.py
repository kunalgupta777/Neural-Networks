## 11.) Implement Heteroassociative Networks
## These are similar to auto associative networks, with the difference that output and input layers can have different number of 
## neurons
print "Hetero Associative Networks"
n,m = map(int, raw_input("Enter n and m:").split())
X = [ random.choice([-1,1]) for i in range(n)]
Y = [ random.choice([-1,1]) for i in range(m)]
print "Input Vector is",X
print "Output Vector is",Y
weights = [ [ 0 for _ in range(m)] for _ in range(n)]
## Training Phase
for i in range(n):
    for j in range(m):
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
for j in range(m):
    yinj = 0
    for i in range(n):
        yinj+=test[i]*weights[i][j]
    yin = f(yinj)
    outs.append(yin)
print "Testing Output",outs