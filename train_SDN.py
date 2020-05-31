import torch
import torch.nn as nn

import torch.optim as optim
import warnings
from net_model import Net
from generate_data import G
import pickle
from tensorboardX import SummaryWriter



writer = SummaryWriter('log')
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 256



#借用tensorboard

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
epoch = 160
step = 600000

def train(step):
    max = 0
    record = 0
    train_record  = 0
    train_time = 0
    for i in range(step):
        _images, _labels = g.train_next_batch(batch_size)
        images = torch.from_numpy(_images).type(torch.float)
        _,labels = torch.max(torch.from_numpy(_labels), 1)
        images = images
        labels = labels
        optimizer.zero_grad()
        outputs, succ, train = net(images)
        goal = criterion(outputs, labels)

        if succ<0.8:
            loss = goal+ 1e-5*train
            train_record += 1e-5*train
            train_time += 1
        else:
            loss = goal
        loss.backward()
        optimizer.step()

        # 记载损失
        info = {'loss': goal}
        for tag, value in info.items():
            writer.add_scalar(tag, value, i)
        record += goal

        if i % 1000 == 0:
            correct = 0
            total = 0
            _test_images = g.test_images[:20000]
            _test_labels = g.test_labels[:20000]
            test_images = torch.from_numpy(_test_images).type(torch.float)
            _,test_labels =  torch.max(torch.from_numpy(_test_labels), 1)
            outputs, succ, _ = net(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
            accuracy_value = correct/total
            judge_result,_,_ = net(torch.tensor([[0,0]]).float())
            if accuracy_value>max and judge_result[0][0]>judge_result[0][1]:
                max = accuracy_value
                torch.save(net, 'model/net.pkl')
                with open('output.txt', 'a') as f:
                    print('max', file=f)
            with open('output.txt', 'a') as f:
                print('step:%d accuracy:%.4f  loss:%.4f train:%.6f succ:%.4f' % (i, accuracy_value, record / 1000, (train_record/train_time) if train_time !=0 else 0, succ),file=f)
            record = 0
            train_record = 0
            train_time = 0
if __name__ == '__main__':
    _temp_obj = open("data", "rb")
    g = pickle.load(_temp_obj)
    _temp_obj.close()
    train(step)
