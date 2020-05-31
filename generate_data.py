import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class G:
    def __init__(self):
        self.dim = 2
        self.center = [185]*self.dim
        _temp_train1 = np.random.uniform(low=0, high=255, size=(120000, self.dim))
        _temp_train2 = np.random.uniform(low=[0 for i in range(self.dim)],
                                         high=[4 for i in range(self.dim)], size=(36000, self.dim))
        self.train = np.concatenate([_temp_train1[:int(len(_temp_train1)/6*5)], _temp_train2[:int(len(_temp_train2)/6*5)]], axis=0)
        self.test_images = np.concatenate([_temp_train1[int(len(_temp_train1)/6*5):], _temp_train2[int(len(_temp_train2)/6*5):]], axis=0)
        self.train_labels = []
        for i in self.train:
            if self._judge(i):
                self.train_labels.append([1,0])
            else:
                self.train_labels.append([0,1])
        self.train_labels = np.array(self.train_labels)
        self.test_labels = []
        for i in self.test_images:
            if self._judge(i):
                self.test_labels.append([1, 0])
            else:
                self.test_labels.append([0, 1])
        self.test_labels = np.array(self.test_labels)
        self.train=self.train/255
        self.test_images=self.test_images/255
        print("finish G")

    def _judge(self,input):
        sum = 0
        _temp1 = True
        for i in range(self.dim):
            sum+=input[i]
            if abs(input[i]-self.center[i])>=70:
                _temp1 = False

        if sum<10:
            return True
        elif _temp1:
            return True
        else:
            return False

    def train_next_batch(self, num):
        _label = np.random.randint(low=0, high=len(self.train), size=num)
        _return_label = []
        _return_image = []
        for i in _label:
            _return_label.append(self.train_labels[i])
            _return_image.append(self.train[i])
        _return_image = np.array(_return_image)
        _return_label = np.array(_return_label)
        return _return_image, _return_label
if __name__ == "__main__":
    g = G()
    _temp_obj = pickle.dumps(g)
    with open("data", "wb")as mnist:
        mnist.write(_temp_obj)
    temp = []
    for i in range(len(g.train)):
        if g.train_labels[i][0] == 1:
            temp.append(g.train[i])
    new = np.array(temp)
    plt.scatter(new[...,0], new[...,1],0.5,marker=".")


    '''ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度

    ax.scatter(new[..., 0], new[..., 1], new[..., 2], c='g', marker=".")
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')'''
    plt.show()