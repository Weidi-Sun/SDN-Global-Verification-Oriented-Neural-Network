import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import multiprocessing
import warnings
import pickle

warnings.filterwarnings("ignore")
import time
import sys
from net_model import Net

import copy
import matplotlib.pyplot as plt


class ruleTree:
    def __init__(self, index=0):
        self.index = index
        self.child = []
        self.v = None
        self.connect = None
        self.contain = None
        self.center = None
        self.r = None
        self.R = None
        self.searched = False

    def search(self, num):
        def binary_search(list, num, scope):
            temp = int((scope[0] + scope[1]) / 2)
            if scope[1] - scope[0] == 1:
                return temp
            elif list[temp - 1].index < num and num <= list[temp].index:
                return temp
            elif list[temp].index < num:
                return binary_search(list, num, (temp + 1, scope[1]))
            elif num <= list[temp - 1].index:
                return binary_search(list, num, (scope[0], temp))

        if num > self.index:
            raise ValueError("Overflow")
        else:
            temp_index = binary_search(self.child, num, (0, len(self.child)))
            if self.child[temp_index].index == num and not isinstance(self.child[temp_index].child[0], ruleTree):
                return self.child[temp_index]
            else:
                return self.child[temp_index].search(num)


class treeStack:
    def __init__(self):
        self.list = []

    def push(self, e):
        self.list.append(e)

    def pop(self):
        if self.list != []:
            return self.list.pop()
        else:
            print("stack is empty")
            raise ValueError

    def top(self):
        if self.list != []:
            return self.list[-1]
        else:
            print("stack is empty")
            raise ValueError


def ActiveRule(net, layer, temp, active_group):
    if active_group != None:
        for i in range(int(active_group[0]), int(active_group[1])):
            temp.child[0].append(net[layer][i])


def InActiveRule(net, layer, temp, inactive_group):
    if inactive_group != None:
        for i in range(int(inactive_group[0]), int(inactive_group[1])):
            temp.child[0].append((-np.array(net[layer][i])).tolist())


def possibelGroup_to_index(way, g_info):
    index = 0
    for layer in range(len(way)):
        index = index * (g_info[layer][0] * g_info[layer][0] + g_info[layer][0] + 1)
        if way[layer][0] < way[layer][1]:
            index += (way[layer][0] * g_info[layer][0] + way[layer][1] - 1)
        else:
            index += (way[layer][0] * g_info[layer][0] + way[layer][1])
    return index


def index_to_group(index, g_info):
    require = []
    for i in range(len(g_info) - 1):
        _layer_total = (g_info[i][0] * g_info[i][0] + g_info[i][0] + 1)
        _temp = index % _layer_total
        if _temp == _layer_total - 1:
            g = g_info[i][0]
            g_ = g_info[i][0]
        else:
            g = (_temp) // g_info[i][0]
            g_ = ((_temp) % g_info[i][0])
            if g_ >= g:
                g_ += 1
        require.append([g, g_])
        index = index // _layer_total
    return require

def new_index_to_group(index, g_info):
    require = []
    for i in range(len(g_info) - 2,-1,-1):
        _layer_total = (g_info[i][0] * g_info[i][0] + g_info[i][0] + 1)
        _temp = index % _layer_total
        if _temp == _layer_total - 1:
            g = g_info[i][0]
            g_ = g_info[i][0]
        else:
            g = (_temp) // g_info[i][0]
            g_ = ((_temp) % g_info[i][0])
            if g_ >= g:
                g_ += 1
        require = [[g,g_]]+require
        index = index // _layer_total
    return require

def satisfy(cons, y):
    result = []
    for i in cons:
        _temp = np.array(i)
        coefficient = _temp[..., :-1]
        _add = torch.from_numpy(_temp[..., -1]).float()
        coefficient = torch.from_numpy(coefficient.T).float()
        _temp = y.mm(coefficient)

        for t in range(len(y)):
            _temp[t] += _add
        _label_pos = torch.where(_temp >= 0, torch.tensor(1), torch.tensor(0))

        temp_result = torch.sum(_label_pos, dim=-1)

        temp_result = torch.where(temp_result == len(i), torch.tensor(1), torch.tensor(0)).tolist()

        result.append(temp_result)
    '''result = np.sum(np.array(result).T, axis=-1)'''
    result = np.array(result).T
    return result


'''def con_satisfy(cons,y):
    new_cons = cons

    add = new_cons[...,-1]

    mul = new_cons[...,:-1].T

    _temp = np.dot(y,mul)

    for i in range(len(y)):
        _temp[i] = _temp[i]+add


    _temp = np.where(_temp>=0,1,0)

    _temp = np.sum(_temp,axis=-1).reshape([-1])
    print(_temp)
    _temp = np.where(_temp==len(new_cons),True,False)

    return _temp.tolist()'''


def con_satisfy(cons, y):
    add = cons[..., -1]

    mul = cons[..., :-1].T

    _temp = np.dot(y, mul)

    for i in range(len(y)):
        _temp[i] = _temp[i] + add

    _temp = np.where(_temp >= 0, 1, 0)

    _temp = np.sum(_temp, axis=1)

    _temp = np.where(_temp == len(cons), True, False)
    return _temp.tolist()


def shake(classify_cons, node, configs, g_info, ONet, test_num, total_possible):
    upper_b = []
    for i in configs:
        upper_b.append([i.get("min"), i.get("max")])
    bound = np.array(node.child[0])
    for i in range(len(configs)):
        bound[..., i] = np.where(bound[..., i] > 0, configs[i].get("max"), configs[i].get("min"))

    bound[..., -1] = np.array([1] * len(node.child[0]))
    cons = np.array(node.child[0])
    _temp = cons * bound
    _temp_sum = np.sum(_temp, axis=1)

    for i in range(len(configs)):
        reduce = _temp[..., i]
        _temp_i_sum = _temp_sum - reduce
        _temp_cons = cons[..., i]
        for j in range(len(_temp_cons)):
            if _temp_cons[j] == 0:
                continue
            _temp_bound = -_temp_i_sum[j] / _temp_cons[j]
            if _temp_cons[j] < 0:
                if upper_b[i][1] > _temp_bound:
                    upper_b[i][1] = _temp_bound
            else:
                if upper_b[i][0] < _temp_bound:
                    upper_b[i][0] = _temp_bound
    upper_v = 1

    for i in upper_b:
        if i[0] >= i[1]:
            return None, [None], None,None,None,None
        else:
            upper_v *= (i[1] - i[0])
    rand_list = []

    low = []
    high = []
    for i in upper_b:
        low.append(i[0])
        high.append(i[1])
    rand = np.random.uniform(low, high, size=(test_num, len(upper_b)))
    torch_rand = torch.from_numpy(rand).float()
    require = index_to_group(node.index, g_info)
    y, way = ONet.back_forward(torch_rand, g_info)

    num = 0

    contain = []
    '''judge = con_satisfy(np.array(node.child[0]), torch_rand,node)


    for i in range(len(judge)):
        if judge[i]:
            num+=1
            temp_center = temp_center + rand[i]
            contain.append(torch_rand[i].tolist())'''

    target = node.index // total_possible

    compare = satisfy(classify_cons, y)
    for i in range(len(y)):
        _temp_judge = True
        if compare[i][int(target)] == 1:
            if way[i] != require:
                _temp_judge = False
        else:
            _temp_judge = False
        if _temp_judge:
            contain.append(torch_rand[i].tolist())
            num += 1


    child = None
    if num == 0:
        v = None
        contain = None
        r = None
        R = None
        center = None
    elif num == 1:
        child = node.child[0]
        k = 0
        while k < len(child):
            i = k + 1
            while i < len(child):
                if child[i] == child[k]:
                    child.pop(i)
                else:
                    i += 1
            k += 1
        R=1/255*5
        center = np.array(contain[0])
        while True:
            rand = np.random.uniform(center - R, center + R, size=(test_num, len(upper_b)))
            torch_rand = torch.from_numpy(rand).float()
            y, way = ONet.back_forward(torch_rand, g_info)
            compare = satisfy(classify_cons, y)
            contain = []
            num = 0
            for i in range(len(y)):
                _temp_judge = True
                if compare[i][int(target)] == 1:
                    if way[i] != require:
                        _temp_judge = False
                else:
                    _temp_judge = False
                if _temp_judge:
                    contain.append(torch_rand[i].tolist())
                    num += 1
            _contain = np.array(contain)
            center = np.sum(_contain, axis=0) / len(contain)
            new_R = max_dis(center, _contain)
            v = pow(R, len(upper_b)) * num / test_num
            if R - new_R<1/255:
                R+= 1/255
            elif R - new_R>1/255*2:
                R = new_R
            else:
                break
        rand = np.random.uniform(center - R, center + R, size=(test_num, len(upper_b)))
        torch_rand = torch.from_numpy(rand).float()
        y, way = ONet.back_forward(torch_rand, g_info)
        compare = satisfy(classify_cons, y)
        r_contain = []
        num = 0
        for i in range(len(y)):
            if compare[i][int(target)] == 1:
                r_contain.append(torch_rand[i].tolist())
                num += 1
        r = num/test_num

    else:
        child = node.child[0]
        k = 0
        while k < len(child):
            i = k + 1
            while i < len(child):
                if child[i] == child[k]:
                    child.pop(i)
                else:
                    i += 1
            k += 1
        _contain = np.array(contain)
        center = np.sum(_contain, axis=0) / len(contain)
        R = max_dis(center, _contain)
        while True:
            rand = np.random.uniform(center-R, center+R, size=(test_num, len(upper_b)))
            torch_rand = torch.from_numpy(rand).float()
            y, way = ONet.back_forward(torch_rand, g_info)
            compare = satisfy(classify_cons, y)
            contain = []
            num = 0
            for i in range(len(y)):
                _temp_judge = True
                if compare[i][int(target)] == 1:
                    if way[i] != require:
                        _temp_judge = False
                else:
                    _temp_judge = False
                if _temp_judge:
                    contain.append(torch_rand[i].tolist())
                    num += 1
            v = pow(R,len(upper_b)) * num/test_num
            _contain = np.array(contain)
            center = np.sum(_contain, axis=0) / len(contain)
            new_R = max_dis(center, _contain)
            if R - new_R < 1 / 255:
                R += 1 / 255
            elif R - new_R > 1 / 255 * 2:
                R = new_R
            else:
                break
        rand = np.random.uniform(center - R, center + R, size=(test_num, len(upper_b)))
        torch_rand = torch.from_numpy(rand).float()
        y, way = ONet.back_forward(torch_rand, g_info)
        compare = satisfy(classify_cons, y)
        r_contain = []
        num = 0
        for i in range(len(y)):
            if compare[i][int(target)] == 1:
                r_contain.append(torch_rand[i].tolist())
                num += 1
        r = num / test_num
    child = [child]
    return v, child, contain,center,r,R

def max_dis(center, list):
    max = 0
    for i in list:
        _dif = center-i
        dis = np.sqrt(np.sum(_dif*_dif))
        if dis > max:
            max = dis
    return max

def recursion_V(cons, tree_node, scope, ONet, configs, g_info, total_possible, pool_size):
    if scope[1] - scope[0] > 1:
        child_size = (scope[1] - scope[0]) / len(tree_node.child)
        if child_size == 1:
            pool = multiprocessing.Pool()
            for i in range(len(tree_node.child)):
                node = tree_node.child[i]
                '''node.v, node.child, node.contain = shake(cons,node,configs,g_info,ONet,1000)'''
                node.v, node.child, node.contain,node.center,node.r,node.R = pool.apply_async(shake, (
                cons, node, configs, g_info, ONet, 3000, total_possible)).get()
                if (i + 1) % pool_size == 0:
                    pool.close()
                    pool.join()
                    pool = multiprocessing.Pool()
                elif (i + 1) == len(tree_node.child):
                    pool.close()
                    pool.join()
        else:
            for i in range(len(tree_node.child)):
                recursion_V(cons, tree_node.child[i], (scope[0] + i * child_size, scope[0] + (i + 1) * child_size),
                            ONet, configs, g_info, total_possible, pool_size)


def calculate_V(ONet, cons, tree, configs, g_info, cons_size, total_possible, pool_size):
    total = 1
    for i in range(len(g_info) - 1):
        total *= (g_info[i][0] * g_info[i][0] + g_info[i][0] + 1)
    total *= cons_size
    init_range = (0, total)

    recursion_V(cons, tree, init_range, ONet, configs, g_info, total_possible, pool_size)


def Sort(tree, layer, g_info, cons_size):
    total = 1
    for i in range(layer):
        total *= (g_info[i][0] * g_info[i][0] + g_info[i][0] + 1)
    total *= cons_size
    init_range = (0, total)

    def recursion(tree_node, scope):
        tree_node.index = scope[1] - 1
        if scope[1] - scope[0] > 1:
            child_size = (scope[1] - scope[0]) / len(tree_node.child)
            for i in range(len(tree_node.child)):
                recursion(tree_node.child[i], (scope[0] + i * child_size, scope[0] + (i + 1) * child_size))

    recursion(tree, init_range)
def reSort_node(tree, i, g_info,history = []):
    history.append(i)
    node = tree.search(i)
    if node.v!=None:
        way = index_to_group(i,g_info)
        new_index = possibelGroup_to_index(way,g_info)
        new_node = tree.search(new_index)
        if new_index == history[0]:
            start_node = tree.search(history[0])
            record = [start_node.child,start_node.v,start_node.contain]
            start_node.child, start_node.v, start_node.contain = node.child,node.v,node.contain
            start_node.searched = True
            history.pop()
            return record
        elif new_node.v!=None:
            record = reSort_node(tree, new_index, g_info,history)
            if len(history)==1 and record!= [[None],None,None]:
                new_node.child, new_node.v, new_node.contain = record[0],record[1],record[2]
                new_node.searched = True
            elif len(history)==1:
                new_node.child, new_node.v, new_node.contain = node.child, node.v, node.contain
                new_node.searched = True
                node.child, node.v, node.contain = record[0],record[1],record[2]
                history.pop()
            else:
                new_node.child, new_node.v, new_node.contain = node.child, node.v, node.contain
                new_node.searched = True
                history.pop()
                return record
        else:
            new_node.child, new_node.v, new_node.contain = node.child, node.v, node.contain
            new_node.searched = True
            history.pop()
            return [[None],None,None]
    else:
        return None


def preProcess(ONet):
    net = []
    Weights = True
    for i in ONet.named_parameters():
        if Weights:
            temp_weight = i[1]
            Weights = False
        else:
            temp_bias = i[1].view(-1, 1)
            layer = torch.cat((temp_weight, temp_bias), 1).tolist()
            net.append(layer)
            Weights = True
    return net


def fix_bringIn(net, target, active, inactive, g_info, g_layer, net_layer):
    '''print(active)
    print(inactive)
    print("--------------------------")'''
    if active != None:
        if inactive != None:
            if active[0] < inactive[0]:
                filter = [g_info[g_layer][3]] * int(active[0]) + \
                         [g_info[g_layer][2]] * int(g_info[g_layer][1]) + [g_info[g_layer][3]] * int(
                    active[1] - active[0] - g_info[g_layer][1]) + \
                         [g_info[g_layer][3]] * int(inactive[0] - active[1]) + \
                         [0] * int(g_info[g_layer][1]) + [g_info[g_layer][3]] * int(
                    inactive[1] - inactive[0] - g_info[g_layer][1]) + \
                         [g_info[g_layer][3]] * int(len(target[0]) - inactive[1] - 1) + [1]
            else:
                filter = [g_info[g_layer][3]] * int(inactive[0]) + \
                         [0] * int(g_info[g_layer][1]) + [g_info[g_layer][3]] * int(
                    inactive[1] - inactive[0] - g_info[g_layer][1]) + \
                         [g_info[g_layer][3]] * int(active[0] - inactive[1]) + \
                         [g_info[g_layer][2]] * int(g_info[g_layer][1]) + [g_info[g_layer][3]] * int(
                    active[1] - active[0] - g_info[g_layer][1]) + \
                         [g_info[g_layer][3]] * int(len(target[0]) - active[1] - 1) + [1]
        else:
            filter = [g_info[g_layer][3]] * int(active[0]) + \
                     [g_info[g_layer][2]] * int(g_info[g_layer][1]) + [g_info[g_layer][3]] * int(
                active[1] - active[0] - g_info[g_layer][1]) + \
                     [g_info[g_layer][3]] * int(len(target[0]) - active[1] - 1) + [1]
    else:
        if inactive != None:
            filter = [g_info[g_layer][3]] * int(inactive[0]) + \
                     [0] * int(g_info[g_layer][1]) + [g_info[g_layer][3]] * int(
                inactive[1] - inactive[0] - g_info[g_layer][1]) + \
                     [g_info[g_layer][3]] * int(len(target[0]) - inactive[1] - 1) + [1]
        else:

            filter = [g_info[g_layer][3]] * (len(target[0]) - 1) + [1]

    filter = torch.from_numpy(np.array([filter] * len(target))).double()
    target = torch.from_numpy(np.array(target)).double()

    para = torch.from_numpy(np.array(net[net_layer])).double()
    weight = (target.mul(filter))[..., :-1]
    result = weight.mm(para)
    result[..., -1] += target[..., -1]
    return result.tolist()


def fix_back(way, net, location, layer, BoL, g_info):
    cons = [[0] * (location) + [BoL] + [0] * (len(net[layer]) - location)]
    cons = fix_bringIn(net, cons, None, None, g_info, -1, layer)
    layer -= 1
    while layer != -1:
        if way[layer][0] != g_info[layer][0]:
            _active_group = (way[layer][0] * g_info[layer][1], way[layer][0] * g_info[layer][1] + g_info[layer][1])
        else:
            _active_group = None
        if way[layer][1] != g_info[layer][0]:
            _inactive_group = (way[layer][1] * g_info[layer][1], way[layer][1] * g_info[layer][1] + g_info[layer][1])
        else:
            _inactive_group = None
        cons = fix_bringIn(net, cons, _active_group, _inactive_group, g_info, layer, layer)
        layer -= 1
    return cons


def bringIn(net, target, active, inactive, g_info, layer):
    if active != None:
        if inactive != None:
            if active[0] < inactive[0]:
                filter = [g_info[layer][3]] * int(active[0]) + \
                         [g_info[layer][2]] * int(g_info[layer][1]) + [g_info[layer][3]] * int(
                    active[1] - active[0] - g_info[layer][1]) + \
                         [g_info[layer][3]] * int(inactive[0] - active[1]) + \
                         [0] * int(g_info[layer][1]) + [g_info[layer][3]] * int(
                    inactive[1] - inactive[0] - g_info[layer][1]) + \
                         [g_info[layer][3]] * int(len(target[0]) - inactive[1] - 1) + [1]
            else:
                filter = [g_info[layer][3]] * int(inactive[0]) + \
                         [0] * int(g_info[layer][1]) + [g_info[layer][3]] * int(
                    inactive[1] - inactive[0] - g_info[layer][1]) + \
                         [g_info[layer][3]] * int(active[0] - inactive[1]) + \
                         [g_info[layer][2]] * int(g_info[layer][1]) + [g_info[layer][3]] * int(
                    active[1] - active[0] - g_info[layer][1]) + \
                         [g_info[layer][3]] * int(len(target[0]) - active[1] - 1) + [1]
        else:
            filter = [g_info[layer][3]] * int(active[0]) + \
                     [g_info[layer][2]] * int(g_info[layer][1]) + [g_info[layer][3]] * int(
                active[1] - active[0] - g_info[layer][1]) + \
                     [g_info[layer][3]] * int(len(target[0]) - active[1] - 1) + [1]
    else:
        if inactive != None:
            filter = [g_info[layer][3]] * int(inactive[0]) + \
                     [0] * int(g_info[layer][1]) + [g_info[layer][3]] * int(
                inactive[1] - inactive[0] - g_info[layer][1]) + \
                     [g_info[layer][3]] * int(len(target[0]) - inactive[1] - 1) + [1]
        else:
            filter = [g_info[layer][3]] * (len(target[0]) - 1) + [1]
    filter = torch.from_numpy(np.array([filter] * len(target))).double()
    target = torch.from_numpy(np.array(target)).double()
    para = torch.from_numpy(np.array(net[layer])).double()
    weight = (target.mul(filter))[..., :-1]
    result = weight.mm(para)
    result[..., -1] += target[..., -1]
    return result.tolist()


def Substitution(net, node, g, g_, g_info, layer):
    Temp = ruleTree()
    group_num = g_info[layer - 1][0]
    group_len = int(len(net[layer - 1]) / group_num)
    if g != g_info[layer - 1][0] and g_ != g_info[layer - 1][0]:
        active_group = (g * group_len, (g + 1) * group_len)
        inactive_group = (g_ * group_len, (g_ + 1) * group_len)
    elif g != g_info[layer - 1][0]:
        active_group = (g * group_len, (g + 1) * group_len)
        inactive_group = None
    elif g_ != g_info[layer - 1][0]:
        active_group = None
        inactive_group = (g_ * group_len, (g_ + 1) * group_len)
    else:
        active_group = None
        inactive_group = None
    Temp.child.append(bringIn(net, node.child[0], active_group, inactive_group, g_info, layer - 1))
    ActiveRule(net, layer - 1, Temp, active_group)
    InActiveRule(net, layer - 1, Temp, inactive_group)
    return Temp


def BMRule(net, node, g_info, layer):
    tasks = []
    _new_child = []
    for g in range(g_info[layer - 1][0] + 1):
        for g_ in range(g_info[layer - 1][0] + 1):
            if g != g_ or (g == g_info[layer - 1][0] and g_ == g_info[layer - 1][0]):
                temp = Substitution(net, node, g, g_, g_info, layer)
                _new_child.append(temp)
                if layer - 1 != 0:
                    tasks.append(temp)
    node.child = _new_child
    return tasks, _new_child


def Refine(cons, tree, configs, g_info, total_possible, ONet):
    for i in range(tree.index + 1):
        node = tree.search(i)
        if node.v != None:
            node.v, node.child, node.contain,node.center,node.r,node.R = shake(cons, node, configs, g_info, ONet, 100000, total_possible)
    return


def MAIN(ONet, net, cons, SDAlayers, g_info, configs, pool_size):
    layer = SDAlayers + 1
    tree = ruleTree()
    tasks = []
    for i in range(len(cons)):
        new_tree = ruleTree()
        tasks.append(new_tree)
        tree.child.append(new_tree)
        new_tree.child = [bringIn(net, cons[i], None, None, g_info, layer - 1)]

    for t in range(SDAlayers):

        layer -= 1
        pool = multiprocessing.Pool()

        task_num = len(tasks)
        temp_tasks = tasks
        tasks = []

        for i in range(1, task_num + 1):
            node = temp_tasks.pop()

            new_task, child = pool.apply_async(BMRule, (net, node, g_info, layer)).get()
            tasks += new_task
            node.child = child
            if i % pool_size == 0:
                pool.close()
                pool.join()
                pool = multiprocessing.Pool()
            elif i == task_num:
                pool.close()
                pool.join()

    with open('output.txt', 'a') as f:
        print('last layer finish', file=f)

    Sort(tree, SDAlayers, g_info, len(cons))
    with open('output.txt', 'a') as f:
        print('sort finish', file=f)

    total_possible = 1
    for i in range(len(g_info) - 1):
        total_possible *= (g_info[i][0] * g_info[i][0] + g_info[i][0] + 1)

    calculate_V(ONet, cons, tree, configs, g_info, len(cons), total_possible, pool_size)
    with open('output.txt', 'a') as f:
        print('calculate_V finish', file=f)

    Refine(cons, tree, configs, g_info, total_possible, ONet)
    with open('output.txt', 'a') as f:
        print('Refine finish', file=f)
    for i in range(tree.index+1):
        node = tree.search(i)
        if node.v!=None and not node.searched:
            record = reSort_node(tree,i,g_info)
            if record!= None:
                node.child, node.v, node.contain = record[0], record[1], record[2]

    for i in range(tree.index+1):
        node = tree.search(i)
        node.searched = False
    _temp_obj = pickle.dumps(tree)
    with open("TREE", "wb")as rule:
        rule.write(_temp_obj)

    return tree


if __name__ == "__main__":
    '''ONet = torch.load('model/net.pkl')
    net = preProcess(ONet)
    with open('output', 'a') as f:
        print('finish_preProcess', file=f)
    cons = [[[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0]
             ]]
    layer = 2
    g_info = [[26, 4, 1, 1],
              [26, 4, 1, 1],
              [1, 1, 1, 1]]
    configs = [{"min": 0, "max": 1} for i in range(16)]
    MAIN(ONet, net, cons, layer, g_info, configs, pool_size=1024)'''
    '''TREE = open("TREE", "rb")
    tree = pickle.load(TREE)
    nodeList = []
    for i in range(tree.index+1):
        node = tree.search(i)
        if node.v!=None:
            nodeList.append(i)
    print(nodeList)'''
    '''MAIN(ONet, net, cons, layer, g_info, configs, pool_size=1024)'''
    '''ONet = test.Net()
    net = [[[0,1,0],[0,1,0],[1,0,1],[1,0,1]],
            [[0,0,1,0,1],[0,0,0,1,1],[1,0,0,0,0],[0,1,0,0,0]],
            [[1, 0,0,0,0], [0, 0,0,1,0]]
           ]
    cons = [[[-1, 1, 0]]
            ]
    layer = 2
    g_info = [
        [2, 2, 1, 1],
        [2, 2, 1, 1],
        [1, 1, 1, 1]
    ]
    configs = [{"min": -4, "max": 4} for i in range(2)]
    tree= MAIN(ONet,net,cons,layer,g_info,configs,pool_size=49)
    for i in range(tree.index+1):
        node = tree.search(i)
        if node.contain!=None:
            _temp = np.array(node.contain)
            picture_x = _temp[..., 0]
            picture_y = _temp[..., 1]
            plt.axis([-4, 4, -4, 4])
            plt.scatter(picture_x, picture_y, 0.5, marker=".")
            plt.savefig("picture/"+"con" + " " + str(i) + " " + str(index_to_group(i, g_info))+".png")
            plt.close()



    net_tree = copy.deepcopy(tree)
    for i in range(net_tree.index+1):
        node = net_tree.search(i)
        node.contain = []
    Test = torch.from_numpy(np.random.uniform(-4, 4, (10000, 2))).type(torch.float)
    y, way = ONet.back_forward(Test, g_info)



    compare = satisfy(cons, y)

    for i in range(10000):
        if compare[i]:
            index = possibelGroup_to_index([0,0,way[i]],g_info)
            node = net_tree.search(index)
            node.contain.append(Test[i].tolist())


    for i in range(net_tree.index+1):
        node = net_tree.search(i)

        if node.contain!=[]:
            temp_way = index_to_group(i,g_info)
            _temp = np.array(node.contain)
            picture_x = _temp[...,0]
            picture_y = _temp[..., 1]
            plt.axis([-4, 4, -4, 4])
            plt.scatter(picture_x, picture_y, 0.5, marker=".")
            plt.savefig("picture/"+"net" + " " + str(i) + " " + str(index_to_group(i, g_info))+".png")
            plt.close()'''

    '''ONet = torch.load('test_model/net.pkl')
    net = preProcess(ONet)
    g_info = [
        [8, 2, 1, 1],
        [8, 2, 1, 1],
        [1, 1, 1, 1]
    ]
    layer = 2
    cons = [[[1, -1, 0]]
            ]
    configs = [{"min": -4, "max": 4} for i in range(2)]
    tree = MAIN(ONet, net, cons, layer, g_info, configs, pool_size=1024)'''
    '''TREE = open("TREE", "rb")
    tree = pickle.load(TREE)'''
    '''for i in range(tree.index + 1):
        node = tree.search(i)
        if node.contain != None:
            _temp = np.array(node.contain)
            picture_x = _temp[..., 0]
            picture_y = _temp[..., 1]
            plt.axis([-4, 4, -4, 4])
            plt.scatter(picture_x, picture_y, 0.5, marker=".")
            plt.savefig("picture/" + "con" + " " + str(i) + " " + str(index_to_group(i, g_info)) + ".png")
            plt.close()'''

    '''_temp = []
    for i in range(tree.index + 1):
        node = tree.search(i)
        if node.contain != None:
            _temp += node.contain
    _temp = np.array(_temp)
    picture_x = _temp[..., 0]
    picture_y = _temp[..., 1]
    plt.axis([-4, 4, -4, 4])
    plt.scatter(picture_x, picture_y, 0.5, marker=".")
    plt.savefig("picture/" + "con" + " " + str(i) + " " + str(index_to_group(i, g_info)) + ".png")
    plt.close()'''

    '''net_tree = copy.deepcopy(tree)
    for i in range(net_tree.index + 1):
        node = net_tree.search(i)
        node.contain = []
    Test = torch.from_numpy(np.random.uniform(-4, 4, (100000, 2))).type(torch.float)
    y, way = ONet.back_forward(Test, g_info)

    compare = satisfy(cons, y)

    for i in range(100000):
        if compare[i]:
            index = possibelGroup_to_index([0, 0, way[i]], g_info)
            node = net_tree.search(index)
            node.contain.append(Test[i].tolist())

    for i in range(net_tree.index + 1):
        node = net_tree.search(i)

        if node.contain != []:
            temp_way = index_to_group(i, g_info)
            _temp = np.array(node.contain)
            picture_x = _temp[..., 0]
            picture_y = _temp[..., 1]
            plt.axis([-4, 4, -4, 4])
            plt.scatter(picture_x, picture_y, 0.5, marker=".")
            plt.savefig("picture/" + "net" + " " + str(i) + " " + str(index_to_group(i, g_info)) + ".png")
            plt.close()'''

    '''ONet = torch.load('test_model/net.pkl')
    net = preProcess(ONet)
    g_info = [
        [8, 2, 1, 1],
        [8, 2, 1, 1],
        [1, 1, 1, 1]
    ]
    layer = 2
    cons = [[[1, -1, 0]]
            ]
    configs = [{"min": -4, "max": 4} for i in range(2)]
    Test = torch.from_numpy(np.random.uniform(-4, 4, (60000, 2))).type(torch.float)
    y, way = ONet.back_forward(Test,g_info)
    result = []
    compare = satisfy(cons, y)
    for i in range(60000):
        index = possibelGroup_to_index(way[i],g_info)
        if compare[i] and (index == 1193 or index == 1233 or index == 1209 or index == 1168 or index == 1339 or index == 1192 or index == 1208):
            result.append(Test[i].tolist())
    _temp = np.array(result)
    picture_x = _temp[..., 0]
    picture_y = _temp[..., 1]
    plt.axis([-4, 4, -4, 4])
    plt.scatter(picture_x, picture_y, 0.5, marker=".")
    plt.savefig("picture/" + "a_test_connect" + ".png")
    plt.close()'''

    '''TREE = open("TREE", "rb")
    tree = pickle.load(TREE)
    L= []
    for i in range(tree.index+1):
        node = tree.search(i)
        if node.child[0]!=None:
            L.append(i)
    print(len(L))'''
