import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


file = 'C:/Users/calwi/OneDrive/Documents/School/Fall17/ECE570_AI/Project/cifar-10-batches-py/data_batch_1'
testfile = 'C:/Users/calwi/OneDrive/Documents/School/Fall17/ECE570_AI/Project/cifar-10-batches-py/test_batch'
#10,000 images in databatch1
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        imgs = pickle.load(fo, encoding='bytes')
    return imgs

imgs = unpickle(file)
test_imgs = unpickle(testfile)
klt,vlt = list(test_imgs.items())[1]
kdt,vdt = list(test_imgs.items())[2]
Xt = vdt #(10000, 3072) pixel values (1 row is 1024 R G B for 1 image)
yt = vlt
#convert back to np arrays
Xt = Xt.astype('float64')
Xt = np.asarray(Xt)
print(Xt)
Xt -= np.mean(Xt, axis = 0)
cov = np.dot(Xt.T,Xt) / Xt.shape[0]
Ut,St,Vt = np.linalg.svd(cov)
Xrott = np.dot(Xt, Ut)
Xt = Xrott / np.sqrt(St + 1e-4) #whitening
print(Xt)
yt = np.asarray(yt)


N = 10000
D = 3072
K = 10

#keys and values for data
kl,vl = list(imgs.items())[1]
kd,vd = list(imgs.items())[2]
kn,vn = list(imgs.items())[3]

#pic = mpimg.imread(vn[0])
#picShow = plt.imshow(pic)

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
for i in range(2000):
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
print(y)
print(np.shape(predicted_class))
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

#Testing with outside data
hidden_layer1t = np.maximum(0, np.dot(Xt, W) + b)
hidden_layer2t = np.maximum(0,np.dot(hidden_layer1t, W2)+b2)
scorest = np.dot(hidden_layer2t, W3) + b3
print(np.shape(scorest))
print(scorest)
predicted_classt = np.argmax(scorest, axis=1)
print(predicted_classt)
print(yt)
print(np.shape(predicted_classt))
print ('training accuracy: %.2f' % (np.mean(predicted_classt == yt)))


