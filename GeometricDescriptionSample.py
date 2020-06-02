import pickle
import torch
import numpy as np
import copy
import multpro_back_pro
from multpro_back_pro import treeStack
from multpro_back_pro import ruleTree
from multpro_back_pro import new_index_to_group as index_to_group
from multpro_back_pro import fix_back
from multpro_back_pro import preProcess
from multpro_back_pro import possibelGroup_to_index
from multpro_back_pro import con_satisfy
from multpro_back_pro import satisfy
from multpro_back_pro import shake
from multpro_back_pro import bringIn
from multpro_back_pro import Substitution
from multpro_back_pro import max_dis
from multpro_back_pro import bound_limit
import multiprocessing
from multiprocessing import Process
from net_model import Net

class group:
    def __init__(self,mem,index):
        self.mem = mem
        self.v = 0
        self.index = index
        self.center = None
        self.R = None


class Result:
    def __init__(self,tree,groupList,pointList,protrudingPoints,smallgroups):
        self.tree = tree
        self.groupList = groupList
        self.pointList = pointList
        self.protrudingPoints = protrudingPoints
        self.smallgroup = smallgroups



def find_upperbound(_temp_convex,configs):
    all_child=copy.deepcopy(_temp_convex)
    upper_b = []
    for i in configs:
        upper_b.append([i.get("min"), i.get("max")])
    bound = np.array(all_child)
    for i in range(len(configs)):
        bound[..., i] = np.where(bound[..., i] > 0, configs[i].get("max"), configs[i].get("min"))
    bound[..., -1] = np.array([1] * len(all_child))
    cons = np.array(all_child)
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

    return upper_b

def generate_possiblePoint(upper_b,_temp_convex,target_con):
    total_step = 100
    upper_b = copy.deepcopy(upper_b)
    temp_convex = copy.deepcopy(_temp_convex)
    pop_con = temp_convex.pop(target_con)
    k = 0
    for i in range(len(pop_con)):
        if np.abs(pop_con[i]) > 1e-4:
            k = i
            break
        elif i == len(pop_con) - 1:
            return False,None

    k = int(k)
    limit = upper_b.pop(k)
    upper_b = np.array(upper_b)
    coe = pop_con.pop(k)
    bias = pop_con.pop()
    reduce = np.array(pop_con)
    try_step = 0


    while try_step<total_step:
        base = np.random.uniform(low=upper_b[...,0].tolist(),high=upper_b[...,1].tolist(),size = (len(pop_con)))

        _temp = (-np.dot(reduce,base)-bias)/coe
        if _temp<=limit[1] and _temp>=limit[0]:
            _result_vector = np.insert(base,k,_temp,0).reshape(1,-1)

            judge = con_satisfy(np.array(temp_convex),_result_vector)
            if judge[0]:
                return True,_result_vector[0].tolist()
        try_step+=1
    return False,None


def judge_target_bound(original_cons,t,ori_index,_temp_node,ONet,g_info,configs,total_possible):
    _temp_convex = _temp_node.child[0]
    upper_b = find_upperbound(_temp_convex, configs)
    k = 0
    step = 1000
    test_size = 100
    connect = set()
    target = _temp_node.index//total_possible
    while k < step:
        valid,_result_vector = generate_possiblePoint(upper_b,_temp_convex,-t)
        _result_vector = np.array(_result_vector)
        if valid:
            test_list = torch.from_numpy(np.random.uniform(bound_limit(configs,_result_vector-(1/255*1e-3)),bound_limit(configs,_result_vector+(1/255*1e-3)),size=[test_size,len(_result_vector)])).float()
            y,  new_way = ONet.back_forward(test_list,g_info)
            compare = satisfy(original_cons, y)

            way_info = index_to_group(ori_index,g_info)
            judge = False
            _connect = set()
            for i in range(len(new_way)):
                if new_way[i]==way_info and compare[i][int(target)]==1:
                    judge = True

                if np.sum(compare[i])>=1:
                    div = 0
                    for j in range(len(compare[i])):
                        if compare[i][j]==1:
                            div = j
                            break
                    index = possibelGroup_to_index(new_way[i],g_info)+total_possible*div
                    if index!=ori_index:
                        _connect.add(index)
            if judge:
                connect = connect.union(_connect)
        k+=1
    return connect

def original_fix_back(way,cons, g_info,net):
    layer = len(g_info)
    tree = ruleTree()
    tasks = []
    for i in range(len(cons)):
        new_tree = ruleTree()
        tasks.append(new_tree)
        tree.child.append(new_tree)
        new_tree.child = [bringIn(net, cons[i], None, None, g_info, layer - 1)]
    for t in range(len(g_info)-1):
        layer -= 1
        g= way[layer-1][0]
        g_ = way[layer-1][1]
        for i in range(len(tree.child)):
            node = tree.child[i]
            tree.child[i] = Substitution(net,node,g,g_,g_info,layer)
    _result = []
    for i in range(len(tree.child)):
        node = tree.child[i]
        _result.append(node.child[0])
    return _result

def ConnectionDescription(original_cons,configs,tree,ONet,net,g_info,pool_size):
    total_num = 0
    for i in range(tree.index+1):
        if tree.search(i).v!=None:
            total_num+=1
    print("total_num",total_num)
    pointList = [None]*(tree.index+1)
    groupList = []

    '''pool = multiprocessing.Pool()'''
    num = 0
    total_possible = 1
    for i in range(len(g_info) - 1):
        total_possible *= (g_info[i][0] * g_info[i][0] + g_info[i][0] + 1)
    i = 0
    while i < tree.index+1:
        _temp_node = tree.search(i)
        if _temp_node.child[0]==None or _temp_node.searched:
            i+=1
            continue
        else:
            _temp_node.searched = True
        num+=1
        connect = set()
        way_info = index_to_group(_temp_node.index,g_info)
        print("percent %.4f"% (num / total_num), len(groupList), way_info)
        all_test = 0
        for j in range(len(way_info)):
            if way_info[j][0]!=g_info[j][0]:
                all_test+=g_info[j][1]
            if way_info[j][1]!=g_info[j][0]:
                all_test+=g_info[j][1]

        for t in range(all_test):
            '''_result = pool.apply_async(judge_target_bound,(original_cons,t,  i, _temp_node,ONet,g_info,configs,total_possible)).get()'''
            _result = judge_target_bound(original_cons,t,  i, _temp_node,ONet,g_info,configs,total_possible)

            connect = connect.union(_result)
        '''if i % pool_size == 0:
            pool.close()
            pool.join()
            pool = multiprocessing.Pool()
        elif i == tree.index:
            pool.close()
            pool.join()'''
        connect = list(connect)
        if _temp_node.connect == None:
            _temp_node.connect = len(connect)
        else:
            _temp_node.connect += len(connect)
        _count = 0
        _min_back = float('inf')
        while _count<len(connect):
            node = tree.search(connect[_count])
            if node.v == None:
                if connect[_count]<_min_back:
                    _min_back = int(connect[_count])
                _temp_way = index_to_group(connect[_count],g_info)
                new_cons = original_fix_back(_temp_way,cons,g_info,net)
                node.child = new_cons
                round = 0
                new_v = None
                while new_v == None and round <20:
                    round+=1
                    new_v,new_child,new_contain ,new_center,new_r,new_R= shake(original_cons, node, configs, g_info, ONet, 1000,total_possible)
                if new_v != None:
                    node.child = new_child
                    node.v = new_v/round
                    node.contain = new_contain
                    node.center = new_center
                    node.r = new_r
                    node.R = new_R
                    node.connect = 1
                else:
                    node.child = [None]
                    _temp_node.connect -=1
                    connect.pop(_count)
                    _count -= 1
            else:
                if node.connect == None:
                    node.connect=1
                else:
                    node.connect += 1
            _count+=1

        if connect == []:
            if pointList[i]==None:
                new_group = group([_temp_node], len(groupList))
                new_group.v = _temp_node.v
                groupList.append(new_group)
                pointList[i]=new_group
        else:
            connect.append(i)
            Groups = set()
            t = 0
            while t < len(connect):
                if pointList[int(connect[t])]!=None:
                    new_index = pointList[int(connect[t])].index
                    Groups.add(new_index)
                    connect.pop(t)
                else:
                    t+=1
            if len(Groups) == 0:
                _temp = []
                new_v = 0
                for t in connect:
                    connect_node = tree.search(t)
                    new_v+=connect_node.v
                    _temp.append(connect_node)
                new_group = group(_temp, len(groupList))
                new_group.v = new_v
                groupList.append(new_group)
                for t in connect:
                    pointList[t] = new_group
            elif len(Groups) == 1:
                group_index = Groups.pop()
                new_nodes = set()
                for t in connect:
                    new_nodes.add(tree.search(t))

                _group = groupList[group_index]
                new_nodes = new_nodes.union(set(_group.mem))
                new_nodes_list = list(new_nodes)
                groupList[group_index].mem = new_nodes_list
                new_v = 0
                for t in new_nodes_list:
                    new_v+=t.v
                    pointList[int(t.index)]= groupList[group_index]
                groupList[group_index].v = new_v
            else:
                temp_groups = list(Groups)
                temp_groups.sort()
                new_nodes = set()
                min_group_index = len(groupList)-1

                for t in connect:
                    new_nodes.add(tree.search(t))
                for t in temp_groups:
                    _group = groupList[t]
                    if _group.index<min_group_index:
                        min_group_index = _group.index
                    new_nodes = new_nodes.union(set(_group.mem))
                new_nodes_list = list(new_nodes)
                groupList[min_group_index].mem = new_nodes_list
                new_v = 0
                for t in new_nodes_list:
                    new_v+=t.v
                    pointList[int(t.index)]= groupList[min_group_index]
                groupList[min_group_index].v = new_v
                delate = 0
                for t in temp_groups[1:]:
                    groupList.pop(t-delate)
                    delate+=1
                for t in range(temp_groups[1],len(groupList)):
                    groupList[t].index = t
        i+=1
        if _min_back != float("inf") and _min_back<i:
            i = _min_back
    _temp_obj = pickle.dumps(tree)
    with open("new_TREE", "wb")as rule:
        rule.write(_temp_obj)
    for _group in groupList:
        center  = None
        total_contain = []
        for index in range(len(_group.mem)):
            vertex = _group.mem[index]
            '''print(vertex.index)
            print(len(vertex.contain))
            print(vertex.v)
            print(vertex.center)
            print(vertex.r)
            print(vertex.R)'''
            if index == 0:
                center = vertex.center*vertex.v
            else:
                center += vertex.center*vertex.v
            total_contain += vertex.contain
        center = center/_group.v
        _group.center = center
        total_contain = np.array(total_contain)
        R = max_dis(center,total_contain)
        _group.R = R
    return tree,groupList,pointList

def select(R,r, groupList ,pointList):
    small_size_isolated = []
    protruding_region = []
    for i in groupList:
        if i.R<R:
            small_size_isolated.append(i)
        else:
            for j in i.mem:
                if j.r!=None and j.r<r:
                    protruding_region.append(j)
    return small_size_isolated,protruding_region


if __name__ == "__main__":
    ONet = torch.load('model/net.pkl')
    net = multpro_back_pro.preProcess(ONet)
    with open('output.txt', 'a') as f:
        print('finish_preProcess', file=f)
    '''cons = [[[-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0]
             ]]
    layer = 2
    g_info = [[20,4,2,1],
              [20,4,2,1],
              [1,1,1,1]]
    configs = [{"min": 0, "max": 1} for i in range(16)]
    TREE = open("TREE", "rb")
    tree = pickle.load(TREE)'''
    '''num =0
    for i in range(tree.index+1):
        node = tree.search(i)
        if node.child[0]!=None:
            print(index_to_group(i,g_info))
            num+=1
            print(node.child[0])
            print(node.v)
    print(num)'''
    '''ONet = test.Net()
    net = [ [[0,1,0],[0,1,0],[1,0,1],[1,0,1]],
            [[0,0,1,0,1],[0,0,0,1,1],[1,0,0,0,0],[0,1,0,0,0]],
            [[1, 0,0,0,0], [0, 0,0,1,0]]
            ]
    cons = [[[-1,1,0]]]
    layer = 2
    g_info = [
          [2,2,1,1],
          [2,2,1,1],
          [1,1,1,1]
          ]
    configs = [{"min": -4, "max": 4} for i in range(2)]'''
    '''tree = multpro_back_pro.MAIN(ONet,net,cons,layer,g_info,configs,pool_size=1024)'''

    '''ONet = torch.load('test_model/net.pkl')
    net = preProcess(ONet)'''
    g_info = [
        [16, 2, 2, 1],
        [16, 2, 2, 1],
        [1, 1, 1, 1]
    ]
    layer = 2
    cons = [[[1, -1, 0]]
            ]
    configs = [{"min": 0, "max": 1} for i in range(2)]
    '''TREE = open("rule_TREE", "rb")
    tree = pickle.load(TREE)'''
    tree = multpro_back_pro.MAIN(ONet, net, cons, layer, g_info, configs, pool_size=1024)
    tree, groupList ,pointList=  ConnectionDescription(cons,configs,tree,ONet,net,g_info,pool_size=1024)
    small_size_isolated_connected_component, protruding_region = select(0.1,0.2, groupList,pointList)
    print(small_size_isolated_connected_component)
    print(protruding_region)
    with open('output.txt', 'a') as f:
        print('description finished', file=f)

    min = float("inf")

    def Vsort(obj):
        return obj.v
    groupList.sort(key=Vsort)
    min_groups = groupList[:int(len(groupList)/10+1)]


    _temp_list = []
    for leaf in range(tree.index+1):
        node = tree.search(leaf)
        if node.connect!= None:
            _temp_list.append(node)
    def rsort(obj):
        return obj.r
    _temp_list.sort(key=rsort)
    protrudingPoints = _temp_list[:int(len(_temp_list) / 10 + 1)]

    result = Result(tree,groupList,pointList,_temp_list,min_groups)

    with open("result","wb") as R:
        R.write(pickle.dumps(result))

    with open('output.txt', 'a') as f:
        print('all finished', file=f)

