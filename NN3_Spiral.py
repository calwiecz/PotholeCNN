import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
  #print(y)
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])

#NN Code with 1 hidden layer

#init para
h = 100
W = 0.01 * np.random.randn(D,h)
print(np.shape(W))
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,h)
b2 = np.zeros((1,h))
W3 = 0.01 * np.random.randn(h,K)
b3 = np.zeros((1,K))

step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
    # evaluate class scores with a 2-layer Neural Network
    hidden_layer1 = np.maximum(0,np.dot(X,W)+b) #ReLU activation
    #print(np.shape(hidden_layer1))
    #print(np.shape(hidden_layer1))
    hidden_layer2 = np.maximum(0,np.dot(hidden_layer1, W2)+b2)
    #print(np.shape(hidden_layer2))
    scores = np.dot(hidden_layer2, W3) + b3
 
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #Loss: Average cross-entropy & Reg
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    
    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))
    
    #Gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    
    dW3 = np.dot(hidden_layer2.T, dscores)
    db3 = np.sum(dscores, axis=0,keepdims=True)
    
    dhidden2 = np.dot(dscores, W3.T)
    dhidden2[hidden_layer2 <= 0] = 0
    
    dW2 = np.dot(hidden_layer1.T, dhidden2)
    db2 = np.sum(dhidden2, axis=0,keepdims=True)
    
    dhidden1 = np.dot(dhidden2, W2.T)
    
    dhidden1[hidden_layer1 <= 0] = 0
    
    dW = np.dot(X.T, dhidden1)
    db = np.sum(dhidden1, axis=0, keepdims=True)
    #print(np.shape(reg))
    
    
    dW3 += reg * W3
    dW2 += reg * W2
    dW += reg * W
    
    #para changes
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2
    W3 += -step_size * dW3
    b3 += -step_size * db3
     
# evaluate training set accuracy
hidden_layer1 = np.maximum(0, np.dot(X, W) + b)
hidden_layer2 = np.maximum(0,np.dot(hidden_layer1, W2)+b2)
scores = np.dot(hidden_layer2, W3) + b3
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0,np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2), W3) + b3
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    


