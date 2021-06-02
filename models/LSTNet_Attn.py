from models.LSTNet import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.Ck = args.CNN_kernel
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        
        # head of multihead attention
        self.head = args.attn_head

        # TODO Attn
        self.attn = nn.MultiheadAttention()


        if(self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        self.output = None

        if(args.output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if(args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m);      
        c = F.relu(self.conv1(c));              
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        for i in range():
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))


        # TODO Attn

