import torch.nn as nn
class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class FPRED_net(nn.Module):
    def __init__(self):
        super(FPRED_net, self).__init__()
        self.model=nn.Sequential(
                nn.Conv3d(1,24,(3,3,3)),
                nn.BatchNorm3d(24),
                nn.ReLU(True),
                nn.Conv3d(24,24,(3,3,3)),
                nn.BatchNorm3d(24),
                nn.ReLU(True),
                nn.MaxPool3d((2,2,2)),
                nn.Dropout3d(),

                nn.Conv3d(24,64,(3,3,3)),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.Conv3d(64,64,(3,3,3)),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d((2,2,2)),
                nn.Dropout3d(),

                nn.Conv3d(64,128,(3,3,3)),
                nn.BatchNorm3d(128),
                nn.ReLU(True),
                nn.Conv3d(128,128,(3,3,3)),
                nn.BatchNorm3d(128),
                nn.ReLU(True),
                nn.MaxPool3d((2,2,2)),
                nn.Dropout3d(),

                Flat(),

                nn.Linear(128,512),
                nn.ReLU(True),
                #nn.Dropout(),
                nn.Linear(512,512),
                nn.ReLU(True),
                #nn.Dropout(),
                nn.Linear(512,2),
                )

    def forward(self, x):
        return self.model(x)
