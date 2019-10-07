import numpy as np

import h5py
filename = 'MNISTdata.hdf5'
f = h5py.File(filename, 'r')
keys = list(f.keys())

test_data = f[keys[0]]
test_label = f[keys[2]]
train_data = f[keys[1]]
train_label = f[keys[3]]


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

class SingleLayerNN:
    def __init__(self,d,d_H,K,learning_rate=1e-2,max_iter=1000000,batch_size=1):
        # input dimension
        self.d = d
        # hidden units number
        self.d_H = d_H
        # output number
        self.K = K
        # learning rate
        self.learning_rate = learning_rate
        # iterations
        self.max_iter = max_iter
        # batch size, maybe 100?
        self.batch_size = batch_size

    def weight_init(self):
        self.W = np.random.random([self.d_H,self.d])/np.sqrt(self.d)
        self.b1 = np.zeros([self.d_H,1])

        self.C = np.random.random([self.K,self.d_H])/np.sqrt(self.d_H)
        self.b2 = np.zeros([self.K,1])

    def forward(self,x):
        Z = np.dot(self.W,x) + self.b1.repeat(x.shape[1]).reshape([self.d_H,x.shape[1]])
        H = ReLU(Z)

        U = np.dot(self.C,H) + self.b2.repeat(x.shape[1]).reshape([self.K,x.shape[1]])
        F = softmax(U)
        return Z,H,U,F

    def backprop(self,H,F,x,y):
        e = np.zeros(F.shape)
        for col in range(y.shape[1]):
            e[y[0][col]][col] = 1
        dRou_dU = -(e-F)
        dRou_db2 = dRou_dU
        dRou_dC = np.dot(dRou_dU,H.T)
        delta = np.dot(self.C.T,dRou_dU)
        dSigma_dZ = np.array(H, copy=True)
        dSigma_dZ[dSigma_dZ>0] = 1
        dRou_db1 = delta*dSigma_dZ
        dRou_dW = np.dot((delta*dSigma_dZ),x.T)
        return dRou_db1,dRou_dW,dRou_dC,dRou_db2

    def objfunc(self,F,y):
        e = np.zeros(F.shape)
        for col in range(y.shape[1]):
            e[y[0][col]][col] = 1

        t = -np.log(F)*e
        return t.sum()

    def test(self,data,label):
        Z,H,U,F = self.forward(data)
        accuracy = 1.0*sum(label[0]==F.argmax(axis=0))/data.shape[1]
        print "Test accuracy:",accuracy
        return accuracy

    def SGD(self,data_train,label_train,data_test,label_test):
        self.weight_init()
        iterations = 0
        data_index = 0
        acc_cnt = 0
        prev_accuracy = 0
        while iterations < self.max_iter:
            # fetch_data
            batch_data = data_train[data_index:data_index+self.batch_size]
            batch_data = batch_data.T
            batch_label = label_train[data_index:data_index+self.batch_size]
            batch_label = batch_label.T

            # forward
            Z,H,U,F = self.forward(batch_data)
            # compute objective function
            obj = self.objfunc(F,batch_label)
            # compute accuracy
            acc_cnt += sum(batch_label[0]==F.argmax(axis=0))
            # backward propagation
            dRou_db1,dRou_dW,dRou_dC,dRou_db2 = self.backprop(H,F,batch_data,batch_label)
            # accumulate update
            dW = dRou_dW/self.batch_size
            db1 = dRou_db1.sum(axis=1).reshape([self.d_H,1])/self.batch_size
            dC = dRou_dC/self.batch_size
            db2 = dRou_db2.sum(axis=1).reshape([self.K,1])/self.batch_size

            self.W -= self.learning_rate*dW/self.batch_size
            self.b1 -= self.learning_rate*db1/self.batch_size
            self.C -= self.learning_rate*dC/self.batch_size
            self.b2 -= self.learning_rate*db2/self.batch_size

            iterations += 1
            data_index = (data_index + self.batch_size)%len(data_train)

            # show accuracy
            if iterations%1000==0:
                train_accuracy = 1.0*acc_cnt/1000/self.batch_size
                print iterations,train_accuracy,obj/self.batch_size
                acc_cnt = 0
                if iterations%10000==0:
                    test_accuracy = self.test(data_test[:].T,label_test[:].T)
                    if abs(prev_accuracy-test_accuracy)>0.97:break
                    prev_accuracy = test_accuracy


if __name__ == "__main__":
    NN = SingleLayerNN(784,100,10)
    NN.SGD(train_data,train_label,test_data,test_label)
