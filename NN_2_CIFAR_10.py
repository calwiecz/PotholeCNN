import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

'''
cifar_train_1 = '../cifar-10-batches-py/data_batch_1'
cifar_train_2 = '../cifar-10-batches-py/data_batch_2'
cifar_train_3 = '../cifar-10-batches-py/data_batch_3'
cifar_train_4 = '../cifar-10-batches-py/data_batch_4'
cifar_train_5 = '../cifar-10-batches-py/data_batch_5'
cifar_test = '../cifar-10-batches-py/test_batch'
'''

#pothole_train_p = np.load('../pothole/Train_data/pothole_data_pos.npy')
#pothole_train_n = np.load('../pothole/Train_data/pothole_data_neg.npy')
'''
pothole_train_p_100 = np.load('../Python_Tut/pothole_data_pos_100.npy')
pothole_train_n_100 = np.load('../Python_Tut/pothole_data_neg_100.npy')
pothole_train_pt = np.load('../pothole/pothole_data_pos_test.npy')
pothole_train_nt = np.load('../pothole/pothole_data_neg_test.npy')
'''
'''
pothole_train_p_100 = np.load('../Python_Tut/pothole_data_pos_100_g.npy')
pothole_train_pt = np.load('../Python_Tut/pothole_data_post_100_g.npy')
pothole_train_n_100 = np.load('../Python_Tut/pothole_data_neg_100_g.npy')
pothole_train_nt = np.load('../Python_Tut/pothole_data_negt_100_g.npy')
'''
pothole_train_p_500 = np.load('../Python_Tut/pothole_data_pos_500_g.npy')
pothole_train_pt = np.load('../Python_Tut/pothole_data_post_500_g.npy')
pothole_train_n_500 = np.load('../Python_Tut/pothole_data_neg_500_g.npy')
pothole_train_nt = np.load('../Python_Tut/pothole_data_negt_500_g.npy')
#remove first row of all pothole data
#pothole_train_p = np.delete(pothole_train_p, (0), axis=0)
#pothole_train_n = np.delete(pothole_train_n, (0), axis=0)
#pothole_train_pt = np.delete(pothole_train_pt, (0), axis=0)
#pothole_train_nt = np.delete(pothole_train_nt, (0), axis=0)

#only 1000 rows of each data
#pothole_train_p_100 = pothole_train_p_100[:10,:]
#pothole_train_n_100 = pothole_train_n_100[:10,:]

train_data = np.vstack((pothole_train_p_500,pothole_train_n_500))
pothole_train_p_100 = None
pothole_train_n_100 = None
test_data = np.vstack((pothole_train_pt,pothole_train_nt))
pothole_train_pt = None
pothole_train_nt = None
print(train_data.shape)
print(test_data.shape)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        imgs = pickle.load(fo, encoding='bytes')
    return imgs
'''
imgs = unpickle(cifar_train_1)
imgs1 = unpickle(cifar_train_1)
imgs2 = unpickle(cifar_train_2)
imgs3 = unpickle(cifar_train_3)
imgs4 = unpickle(cifar_train_4)
imgs5 = unpickle(cifar_train_5)
#print(imgs)
test_imgs = unpickle(cifar_test)
'''
'''
klt,vlt = list(test_imgs.items())[1]
kdt,vdt = list(test_imgs.items())[2]
Xt = vdt #(10000, 3072) pixel values (1 row is 1024 R G B for 1 image)
yt = vlt
#convert back to np arrays
Xt = Xt.astype('float64')
Xt = np.asarray(Xt)
#print(Xt)
Xt -= np.mean(Xt, axis = 0)
cov = np.dot(Xt.T,Xt) / Xt.shape[0]
Ut,St,Vt = np.linalg.svd(cov)
Xrott = np.dot(Xt, Ut)
Xt = Xrott / np.sqrt(St + 1e-4) #whitening
#print(Xt)
yt = np.asarray(yt)
'''
yt = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
o = list(np.zeros((500)))
n = list(np.ones((500)))
yt =np.array(n+o)
yt = list(map(int, yt))
yt = np.array(yt)
Xt = test_data
print(test_data.shape)
N = 200
D = 19200
K = 2

#keys and values for data
'''
kl1,vl1 = list(imgs.items())[1]
kd1,vd1 = list(imgs.items())[2]
kn1,vn1 = list(imgs.items())[3]

kl2, vl2 = list(imgs2.items())[1]
kl3, vl3 = list(imgs3.items())[1]
kl4, vl4 = list(imgs4.items())[1]
kl5, vl5 = list(imgs5.items())[1]

kd2, vd2 = list(imgs2.items())[2]
kd3, vd3 = list(imgs3.items())[2]
kd4, vd4 = list(imgs4.items())[2]
kd5, vd5 = list(imgs5.items())[2]

vl = vl1 + vl2 + vl3 + vl4 + vl5
#vd = vd1 + vd2 + vd3 + vd4 + vd5
vd1 = np.asarray(vd1)
vd2 = np.asarray(vd2)
vd3 = np.asarray(vd3)
vd4 = np.asarray(vd4)
vd5 = np.asarray(vd5)

vd = np.vstack((vd1,vd2,vd3,vd4,vd5))
#print(vl.shape)
print(vd.shape)
#pic = mpimg.imread(vn[0])
#picShow = plt.imshow(pic)
'''
#X = vd #(10000, 3072) pixel values (1 row is 1024 R G B for 1 image)
#y = vl
print(train_data.shape)

X = train_data
train_data = None
#convert back to np arrays
X = X.astype('float64')
X = np.asarray(X)
print(X.shape)
X -= np.mean(X, axis = 0)
X /= 255
'''
cov = np.dot(X.T,X) / X.shape[0]
U,S,V = np.linalg.svd(cov)
cov = None
Xrot = np.dot(X, U)
U = None
X = Xrot / np.sqrt(S + 1e-4) #whitening
Xrot = None
U,S,V = None, None, None
'''
#print(X)
#y = np.asarray(y)
o = list(np.zeros((500)))
n = list(np.ones((500)))
y =np.array(n+o)
y = list(map(int, y))
y = np.array(y)
print(y)
print(y.shape)
#NN Code with 1 hidden layer

#init para
h1 = 60

W = 0.01 * np.random.randn(D,h1)
b = np.zeros((1,h1))
W2 = 0.01 * np.random.randn(h1,K)
b2 = np.zeros((1,K))


iterations = 2000
step_size = 0.5
reg = 1e-3 # regularization strength
dreg = 2
reg_int = 100

print(X.shape)
print(y.shape)

# gradient descent loop
num_examples = X.shape[0]
print(num_examples)
for i in range(iterations):
    # evaluate class scores with a 3-layer Neural Network
    #if (i % reg_int == 0) and (i != 0):
    #    reg /= dreg
    hidden_layer1 = np.maximum(0,np.dot(X,W)+b) #ReLU activation
    scores = np.dot(hidden_layer1, W2) + b2
    
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
    
    dW2 = np.dot(hidden_layer1.T, dscores)
    db2 = np.sum(dscores, axis=0,keepdims=True)
    
    dhidden1 = np.dot(dscores, W2.T)
    
    dhidden1[hidden_layer1 <= 0] = 0
    
    dW = np.dot(X.T, dhidden1)
    db = np.sum(dhidden1, axis=0, keepdims=True)
    

    dW2 += reg * W2
    dW += reg * W
    
    #para changes
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

     
# evaluate training set accuracy
hidden_layer1 = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer1, W2) + b2
print(np.shape(scores))
print(scores)
predicted_class = np.argmax(scores, axis=1)
print(predicted_class)
print(y)
print(np.shape(predicted_class))
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

#Testing with outside data
hidden_layer1t = np.maximum(0, np.dot(Xt, W) + b)
scorest = np.dot(hidden_layer1t, W2) + b2
print(np.shape(scorest))
print(scorest)
predicted_classt = np.argmax(scorest, axis=1)
print(predicted_classt)
print(yt)
print(np.shape(predicted_classt))
print ('training accuracy: %.2f' % (np.mean(predicted_classt == yt)))


