import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)




    def forward(self, x):
        x, succ1, train1 = self.partial_activation(self.fc1(x),16,2,2)
        x, succ2, train2 = self.partial_activation(self.fc2(x),16,2,2)
        x = F.softmax(self.fc3(x))
        succ = (succ1 + succ2 ) / 2
        train = (train1 + train2)
        return x,succ,train

    def back_forward(self, x, g_info):
        layer = 0
        x, succ1, train1, way1 = self.back_partial_activation(self.fc1(x), 16, 2, 2, g_info, layer)
        layer += 1
        x, succ2, train2, way2 = self.back_partial_activation(self.fc2(x), 16, 2, 2, g_info, layer)
        x = F.softmax(self.fc3(x), dim=1)
        succ = (succ1 + succ2) / 2
        train = (train1 + train2)
        way = np.hstack([way1, way2]).reshape([-1, 2, 2])
        return x, way.tolist()

    def partial_activation(self, Y, n, m, t):  # n个门，每个门长m，门占比t
        size = Y.size(0)
        len = Y.size(1)
        block = int(len / n)
        _Y = Y.view([-1, n, block])
        zero_Y = torch.zeros_like(_Y)
        _label_pos = torch.lt(zero_Y, _Y).type(torch.float)
        pos_sum = torch.sum(_label_pos[..., :m], -1)
        minIndex = np.argmin(pos_sum, -1)
        maxIndex = np.argmax(pos_sum, -1)
        train = 0
        succ = 0
        target = []
        for i in range(size):
            _maxIndex = int(maxIndex[i].item())
            _minIndex = int(minIndex[i].item())

            maxSucc = (pos_sum[i][_maxIndex] == m)
            minSucc = (pos_sum[i][_minIndex] == 0)

            '''r = float((torch.div(1, ((total - part))) * ( 1 - t )).cpu().detach().numpy())'''
            q = t  # float((torch.div(1, part) * t).cpu().detach().numpy())
            if maxSucc and minSucc:
                succ += 1
                if _maxIndex < _minIndex:
                    target.append(
                        [1] * block * _maxIndex + [q] * m + [1] * (block - m) + [1] * block * (
                                _minIndex - _maxIndex - 1) + [0.] * m + [1] * (block - m) + [
                            1] * block * (n - _minIndex - 1))
                else:
                    target.append(
                        [1] * block * _minIndex + [0] * m + [1] * (block - m) + [1] * block * (
                                _maxIndex - _minIndex - 1) + [q] * m + [1] * (block - m) + [
                            1] * block * (n - _maxIndex - 1))
            elif maxSucc:
                target.append(
                    [1] * block * _maxIndex + [q] * m + [1] * (block - m) + [1] * block * (n - _maxIndex - 1))
                for k in range(m):
                    if Y[i][_minIndex * m + k] > 0:
                        train += Y[i][_minIndex * block + k]
            elif minSucc:
                target.append(
                    [1] * block * _minIndex + [0] * m + [1] * (block - m) + [1] * block * (n - _minIndex - 1))
                for k in range(m):
                    if Y[i][_maxIndex * m + k] < 0:
                        train -= Y[i][_maxIndex * block + k]
            else:
                for k in range(m):
                    if Y[i][_minIndex * m + k] > 0:
                        train += Y[i][_minIndex * block + k]
                    if Y[i][_maxIndex * m + k] < 0:
                        train -= Y[i][_maxIndex * block + k]
                target.append([1.] * len)

        target = np.array(target)
        target = torch.from_numpy(target).type(torch.float).view(-1, len).detach()
        output = torch.mul(Y, target)
        succ = succ / size

        return output, succ, train

    def back_partial_activation(self, Y, n, m, t, g_info, layer):

        size = Y.size(0)
        len = Y.size(1)
        block = int(len / n)

        zero_Y = torch.zeros_like(Y)
        _label = torch.lt(zero_Y, Y).type(torch.float)
        _Y = Y.view([-1, n, block])
        _label_pos = _label.view([-1, n, block])
        pos_sum = torch.sum(_label_pos[..., :m], -1)
        minIndex = np.argmin(pos_sum, -1)
        maxIndex = np.argmax(pos_sum, -1)
        train = 0
        succ = 0
        target = []
        new_way = []
        for i in range(size):
            _maxIndex = int(maxIndex[i].item())
            _minIndex = int(minIndex[i].item())

            maxSucc = (pos_sum[i][_maxIndex] == m)
            minSucc = (pos_sum[i][_minIndex] == 0)

            '''r = float((torch.div(1, ((total - part))) * ( 1 - t )).cpu().detach().numpy())'''
            q = t  # float((torch.div(1, part) * t).cpu().detach().numpy())
            if maxSucc and minSucc:
                new_way.append([_maxIndex, _minIndex])
                succ += 1
                if _maxIndex < _minIndex:
                    target.append(
                        [1] * block * _maxIndex + [q] * m + [1] * (block - m) + [1] * block * (
                                _minIndex - _maxIndex - 1) + [0.] * m + [1] * (block - m) + [
                            1] * block * (n - _minIndex - 1))
                else:
                    target.append(
                        [1] * block * _minIndex + [0] * m + [1] * (block - m) + [1] * block * (
                                _maxIndex - _minIndex - 1) + [q] * m + [1] * (block - m) + [
                            1] * block * (n - _maxIndex - 1))
            elif maxSucc:
                new_way.append([_maxIndex, g_info[layer][0]])
                target.append(
                    [1] * block * _maxIndex + [q] * m + [1] * (block - m) + [1] * block * (n - _maxIndex - 1))
                for k in range(m):
                    if Y[i][_minIndex * m + k] > 0:
                        train += Y[i][_minIndex * block + k]
            elif minSucc:
                new_way.append([g_info[layer][0], _minIndex])
                target.append(
                    [1] * block * _minIndex + [0] * m + [1] * (block - m) + [1] * block * (n - _minIndex - 1))
                for k in range(m):
                    if Y[i][_maxIndex * m + k] < 0:
                        train -= Y[i][_maxIndex * block + k]
            else:
                new_way.append([g_info[layer][0], g_info[layer][0]])
                for k in range(m):
                    if Y[i][_minIndex * m + k] > 0:
                        train += Y[i][_minIndex * block + k]
                    if Y[i][_maxIndex * m + k] < 0:
                        train -= Y[i][_maxIndex * block + k]
                target.append([1.] * len)

        target = np.array(target)
        target = torch.from_numpy(target).type(torch.float).view(-1, len).detach()
        output = torch.mul(Y, target)
        succ = succ / size

        return output, succ, train, np.array(new_way)

    def new_back_forward(self, x, g_info):
        layer = 0
        x1, succ1, train1, way1,label1 = self.new_back_partial_activation(self.fc1(x), 16, 2, 2, g_info, layer)
        layer += 1
        x2, succ2, train2, way2,label2 = self.new_back_partial_activation(self.fc2(x1), 16, 2, 2, g_info, layer)
        x = F.softmax(self.fc3(x2), dim=1)
        succ = (succ1 + succ2) / 2
        train = (train1 + train2)
        way = np.hstack([way1, way2]).reshape([-1, 2, 2])
        activation_pattern = np.concatenate([label1.reshape(-1,1,len(label1[0])),label2.reshape(-1,1,len(label2[0]))],axis=1)
        return x, way.tolist(),activation_pattern

    def new_back_partial_activation(self, Y, n, m, t, g_info, layer):

        size = Y.size(0)
        len = Y.size(1)
        block = int(len / n)

        zero_Y = torch.zeros_like(Y)
        _label = torch.lt(zero_Y, Y).type(torch.float)
        _Y = Y.view([-1, n, block])
        _label_pos = _label.view([-1, n, block])
        pos_sum = torch.sum(_label_pos[..., :m], -1)
        minIndex = np.argmin(pos_sum, -1)
        maxIndex = np.argmax(pos_sum, -1)
        train = 0
        succ = 0
        target = []
        new_way = []
        for i in range(size):
            _maxIndex = int(maxIndex[i].item())
            _minIndex = int(minIndex[i].item())

            maxSucc = (pos_sum[i][_maxIndex] == m)
            minSucc = (pos_sum[i][_minIndex] == 0)

            '''r = float((torch.div(1, ((total - part))) * ( 1 - t )).cpu().detach().numpy())'''
            q = t  # float((torch.div(1, part) * t).cpu().detach().numpy())
            if maxSucc and minSucc:
                new_way.append([_maxIndex, _minIndex])
                succ += 1
                if _maxIndex < _minIndex:
                    target.append(
                        [1] * block * _maxIndex + [q] * m + [1] * (block - m) + [1] * block * (
                                _minIndex - _maxIndex - 1) + [0.] * m + [1] * (block - m) + [
                            1] * block * (n - _minIndex - 1))
                else:
                    target.append(
                        [1] * block * _minIndex + [0] * m + [1] * (block - m) + [1] * block * (
                                _maxIndex - _minIndex - 1) + [q] * m + [1] * (block - m) + [
                            1] * block * (n - _maxIndex - 1))
            elif maxSucc:
                new_way.append([_maxIndex, g_info[layer][0]])
                target.append(
                    [1] * block * _maxIndex + [q] * m + [1] * (block - m) + [1] * block * (n - _maxIndex - 1))
                for k in range(m):
                    if Y[i][_minIndex * m + k] > 0:
                        train += Y[i][_minIndex * block + k]
            elif minSucc:
                new_way.append([g_info[layer][0], _minIndex])
                target.append(
                    [1] * block * _minIndex + [0] * m + [1] * (block - m) + [1] * block * (n - _minIndex - 1))
                for k in range(m):
                    if Y[i][_maxIndex * m + k] < 0:
                        train -= Y[i][_maxIndex * block + k]
            else:
                new_way.append([g_info[layer][0], g_info[layer][0]])
                for k in range(m):
                    if Y[i][_minIndex * m + k] > 0:
                        train += Y[i][_minIndex * block + k]
                    if Y[i][_maxIndex * m + k] < 0:
                        train -= Y[i][_maxIndex * block + k]
                target.append([1.] * len)

        target = np.array(target)
        target = torch.from_numpy(target).type(torch.float).view(-1, len).detach()
        output = torch.mul(Y, target)
        succ = succ / size

        return output, succ, train, np.array(new_way),_label.numpy()