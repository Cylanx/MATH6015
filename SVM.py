import numpy as np
import cvxopt
from struct import unpack

def readimage(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img
 
def readlabel(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab

#加载数据
def load_data(train_size,digit_combination=[4,9]):
    train_data  = readimage("train-images-idx3-ubyte")[:train_size]
    train_label = readlabel("train-labels-idx1-ubyte")[:train_size]
    test_data   = readimage("t10k-images-idx3-ubyte")
    test_label  = readlabel("t10k-labels-idx1-ubyte")

    def label_data(data,label,digit):
        pos_idx = np.where(label==digit[0])
        neg_idx = np.where(label==digit[1])
        idx = np.concatenate((pos_idx,neg_idx),axis=1)
        out_data = data[idx]
        out_label = np.concatenate((np.ones_like(pos_idx),-np.ones_like(neg_idx)),axis=1)
        return out_data.squeeze(),out_label.squeeze()
    
    train_data, train_label = label_data(train_data,train_label,digit_combination)
    train_data = train_data/255

    test_data, test_label = label_data(test_data,test_label,digit_combination)
    test_data = test_data/255

    return train_data, train_label, test_data, test_label

# 线性核
def linear_kernal(**kwargs):
    def K(x1, x2):
        return np.inner(x1, x2) 
    return K

# 多项式核
def polynomial_kernal(power, coef, **kwargs):
    def K(x1, x2):
        return (np.inner(x1, x2) + coef) ** power
    return K

# 径向基核
def rbf_kernal(gamma, **kwargs):
    def K(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return K

class svm:
    # param kernal:核函数
    # param penaltyC:软间隔惩罚项C
    # param power, gamma,coef：超参数
    def __init__(self, kernal=linear_kernal, penaltyC=1, power=1, gamma=1, coef=1):
        self.kernal = kernal
        self.penaltyC = penaltyC
        self.power = power 
        self.gamma = gamma
        self.coef = coef
        self.kernal = self.kernal(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)
        
    def train(self, x, y):
        x_num = x.shape[0] 
        kernal_matrix = self.kernal(x, x) + (1 / self.penaltyC) * np.eye(x_num)
        
        # 计算凸二次规划
        p = cvxopt.matrix(kernal_matrix * np.outer(y, y))  
        q = cvxopt.matrix(-np.ones([x_num, 1], np.float64))
        g = cvxopt.matrix(-np.eye(x_num))
        h = cvxopt.matrix(np.zeros([x_num, 1], np.float64))
        
        y = np.float64(y)
        a = cvxopt.matrix(y, (1, x_num))
        b = cvxopt.matrix(0.)
        
        # 使用凸规划工具包cvxopt求解SVM目标函数（算lagrange乘子）
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(p, q, g, h, a, b)
        alpha = np.float32(np.array(solution['x']))
        alpha[alpha <= 1e-4] = 0
        
        # 求权重w和截距b
        w = np.sum(np.reshape(y, [-1, 1]) * alpha * x, axis=0)
        b = np.mean(np.reshape(y, [-1, 1]) - np.reshape(np.dot(w, np.transpose(x)), [-1, 1]))
        self.w = w
        self.b = b
        
        return w, b, alpha
    
    def predict(self, x_test):
        y_predict = []
        for sample in x_test:
            predict1 = self.kernal(self.w, sample) + self.b
            predict1 = np.int64(np.sign(predict1))
            y_predict.append(predict1)
        return y_predict

def calc_acc(label,predict):
    return np.sum(label==predict)/len(label)

if __name__ == '__main__':
    digit_combination = [0,1] #[4,9],[4,6],[0,1],[2,7]
    svm = svm(linear_kernal, penaltyC=1, power=1, gamma=1, coef=1)
    train_data, train_label, test_data, test_label = load_data(train_size=10000,digit_combination=digit_combination)
    w, b, alpha = svm.train(x=train_data, y=train_label)
    predict = svm.predict(test_data)
    print('Accuracy:', calc_acc(test_label,predict))





