import numpy as np

import h5py



def ReLU(x):
    return np.maximum(x,0)

def softmax(x):
    exp = np.exp(x)
    try:
        v = exp/sum(exp)
    except e:
        print exp
        exit(-1)
    return v

def img2col(x,k_x,k_y):
    # map each submatrix to a column
    img_col = []
    for i in range(0, x.shape[0] - k_y + 1):
        for j in range(0, x.shape[1] - k_x + 1):
            col = x[i:i + k_y, j:j + k_x].reshape([-1])
            img_col.append(col)
    img_col = np.array(img_col)
    return img_col

def conv(x,kernel):
    k_y,k_x,C = kernel.shape
    col_kernel = kernel.reshape([-1, C])
    col_img = img2col(x,k_x,k_y)
    Z = np.dot(col_img,col_kernel)
    Z = Z.reshape([x.shape[0]-k_y+1,x.shape[1]-k_x+1,C])
    return Z


class SingleLayerCNN:
    def __init__(self,d,k_x,k_y,C,K,learning_rate=1e-2,max_iter=1000000):
        # input dimension 28
        self.d = d
        # convoltion kernel size,number
        self.k_x = k_x
        self.k_y = k_y
        self.C = C
        # output number
        self.K = K
        # learning rate
        self.learning_rate = learning_rate
        # iterations
        self.max_iter = max_iter

    def weight_init(self):
        self.kernel = np.random.random([self.k_y,self.k_x,self.C])/np.sqrt(self.k_y*self.k_x*self.C)

        self.W = np.random.random([self.K,self.d-self.k_y+1,self.d-self.k_x+1,self.C])/np.sqrt(self.K*(self.d-self.k_y+1)*(self.d-self.k_x+1)*self.C)
        self.b = np.zeros([self.K,1])

    def test(self,data,label):
        cnt = 0
        for x,y in zip(data,label):
            Z,H,F = self.forward(x.reshape([self.d,self.d]))
            cnt += sum(y[0]==F.argmax(axis=0))
        accuracy = 1.0*cnt/data.shape[0]
        print "Test accuracy:",accuracy
        return accuracy

    def forward(self,x):
        Z = conv(x,self.kernel)

        H = ReLU(Z)

        U = np.array([(self.W[i]*H).sum() for i in range(self.K)]).reshape([self.K,1])

        F = softmax(U)
        return Z,H,F

    def objfunc(self,F,y):
        e = np.zeros(F.shape)
        for col in range(y.shape[1]):
            e[y[0][col]][col] = 1

        t = -np.log(F)*e
        return t.sum()

    def backprop(self,x,y,Z,H,F):
        e = np.zeros(F.shape)
        for col in range(y.shape[1]):
            e[y[0][col]][col] = 1
        dRou_dU = -(e-F)
        delta = np.dot(dRou_dU.T,self.W.reshape([self.K,-1])).reshape([self.d-self.k_y+1,self.d-self.k_x+1,self.C])
        dSigma_dZ = np.array(H, copy=True)
        dSigma_dZ[dSigma_dZ>0] = 1
        dRou_dK = np.zeros([self.k_y,self.k_x,self.C])
        for c in range(self.C):
            dRou_dK[:,:,c] = conv(x,(dSigma_dZ[:,:,c]*delta[:,:,c]).reshape([self.d-self.k_y+1,self.d-self.k_x+1,1])).reshape([self.k_y,self.k_x])
        dRou_db = dRou_dU
        dRou_dW = np.array([dRou_dU[k]*H for k in range(self.K)])
        return dRou_dK,dRou_db,dRou_dW

    def SGD(self,data_train,label_train,data_test,label_test):
        self.weight_init()
        iterations = 0
        data_index = 0
        acc_cnt = 0
        prev_accuracy = 0
        while iterations < self.max_iter:
            # fetch_data
            batch_data = data_train[data_index:data_index+1]
            batch_data = np.array(batch_data).reshape([self.d,self.d])
            batch_label = label_train[data_index:data_index+1]
            batch_label = batch_label.T

            # forward
            Z,H,F = self.forward(batch_data)

            # objective
            obj = self.objfunc(F,batch_label)
            # compute accuracy
            acc_cnt += sum(batch_label[0]==F.argmax(axis=0))

            # backprop
            dRou_dK,dRou_db,dRou_dW = self.backprop(batch_data,batch_label,Z,H,F)

            # update network
            self.kernel -= self.learning_rate*dRou_dK
            self.W -= self.learning_rate*dRou_dW
            self.b -= self.learning_rate*dRou_db

            # update iterations
            iterations += 1
            data_index = (data_index + 1)%len(data_train)

            # show accuracy
            if iterations%100==0:
                train_accuracy = 1.0*acc_cnt/100
                print iterations,train_accuracy,obj
                acc_cnt = 0
                if iterations%2000==0:
                    test_accuracy = self.test(data_test,label_test)
                    if abs(prev_accuracy-test_accuracy)<0.001:break
                    prev_accuracy = test_accuracy


if __name__ == "__main__":
    filename = 'MNISTdata.hdf5'
    f = h5py.File(filename, 'r')
    keys = list(f.keys())
    test_data = f[keys[0]]
    test_label = f[keys[2]]
    train_data = f[keys[1]]
    train_label = f[keys[3]]

    NN = SingleLayerCNN(28,2,2,16,10)
    NN.SGD(train_data,train_label,test_data,test_label)
