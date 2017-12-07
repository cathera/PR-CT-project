from model import FPRED_net
import torch
from dataset import FPRED_datasets
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader

train_set=FPRED_datasets('./samples_positive_transpose/', './samples_negative_scale1/',1,10000)
test_set=FPRED_datasets('./samples_positive_test/', './samples_negative_test/',5,500)
train_load=DataLoader(train_set, batch_size=16, shuffle=True)
test_load=DataLoader(test_set, batch_size=4, shuffle=True)

model=FPRED_net().cuda()
optimizer=optim.SGD(model.parameters(), lr=0.01)
def train(epoch):
    model.train()
    for batch_id, (data, label) in enumerate(train_load):
        data, label = Variable(data.cuda()), Variable(label.cuda())
        optimizer.zero_grad()
        out=model(data)
        loss=nn.functional.cross_entropy(out, label, weight=torch.Tensor([1,1.5]).cuda())
        loss.backward()
        optimizer.step()
        if batch_id%10==0:
            label_=label==0
            pred=out.data.max(1,keepdim=True)[1].squeeze()
            print((pred & label.data).sum(),label.data.sum())
            print(((pred==0)&label_.data).sum(),label_.data.sum())
            print(epoch, loss.data[0])

def test():
    model.eval()
    test_loss=0
    correct_1=0
    count_1=0
    correct_0=0
    count_0=0
    for data, label in test_load:
        data, label=Variable(data.cuda(), volatile=True), Variable(label.cuda())
        label_=label==0
        out=model(data)
        test_loss+=nn.functional.cross_entropy(out, label).data[0]
        pred=out.data.max(1,keepdim=True)[1].squeeze()
        correct_1+=(pred & label.data).sum()
        count_1+=label.data.sum()
        correct_0+=((pred==0)&label_.data).sum()
        count_0+=label_.data.sum()
    test_loss/=len(test_load.dataset)
    print(correct_0, correct_1, count_0, count_1)
