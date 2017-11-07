import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


file = 'C:/Users/Jesse/Desktop/cifar-10-batches-py/data_batch_1'

#10,000 images in databatch1
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        imgs = pickle.load(fo, encoding='bytes')
    return imgs

imgs = unpickle(file)

#keys and values for data
kl,vl = list(imgs.items())[1]
kd,vd = list(imgs.items())[2]
kn,vn = list(imgs.items())[3]
print('--')
print(vd[0]) #data for each Input
print(vl[0]) #Label for each Input
print(vn[0]) #Filename for each Input
#pic = mpimg.imread(vn[0])
#picShow = plt.imshow(pic)

'''
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
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
'''
N = 10000
D = 3072
K = 10
X = vd #(10000, 3072) pixel values (1 row is 1024 R G B for 1 image)
y = vl
#convert back to np arrays
X = X.astype('float64')
X = np.asarray(X)
print(X)
X -= np.mean(X, axis = 0)
cov = np.dot(X.T,X) / X.shape[0]
U,S,V = np.linalg.svd(cov)
Xrot = np.dot(X, U)
X = Xrot / np.sqrt(S + 1e-4) #whitening
print(X)
y = np.asarray(y)

#NN Code with 1 hidden layer

#init para
h = 200
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,h)
b2 = np.zeros((1,h))
W3 = 0.01 * np.random.randn(h,K)
b3 = np.zeros((1,K))

#step_size = 1e-0
step_size = 5
reg = 1e-3 # regularization strength

print(X.shape)
print(y.shape)

# gradient descent loop
num_examples = X.shape[0]
print(num_examples)
for i in range(1000):
    # evaluate class scores with a 3-layer Neural Network
    
    hidden_layer1 = np.maximum(0,np.dot(X,W)+b) #ReLU activation
    hidden_layer2 = np.maximum(0,np.dot(hidden_layer1, W2)+b2)
    scores = np.dot(hidden_layer2, W3) + b3
 
    exp_scores = np.exp(scores - np.max(scores))
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
print(np.shape(scores))
print(scores)
predicted_class = np.argmax(scores, axis=1)
print(predicted_class)
print(np.shape(predicted_class))
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))




