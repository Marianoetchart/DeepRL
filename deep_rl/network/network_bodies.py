#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DRQNBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DRQNBody, self).__init__()
        self.feature_dim = 512
        self.rnn_input_dim = 7*7*64
        self.batch_size = 1
        self.unroll  = 10 
        in_channels = 1 # for 1 frame input
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.lstm = nn.LSTM(self.rnn_input_dim, self.feature_dim , num_layers = 1)
        self.hidden = self.init_hidden()
        self.reset_flag = False

    def init_hidden(self): 
        # initializing the hidden and cell states
        if str(Config.DEVICE) == 'cpu':
            return (autograd.Variable(torch.zeros(1, 1,self.feature_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.feature_dim)))
        else: 
            return (autograd.Variable(torch.zeros(1, 1,self.feature_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.feature_dim)).cuda())

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(state) for state in h)

    def forward(self, x):
        if self.reset_flag:
            self.hidden = self.repackage_hidden(self.hidden)
            self.reset_flag = False
        
        self.init_hidden()

        ycat = torch.Tensor()
        xchunks= torch.chunk(x,self.unroll, 1)
        output = torch.Tensor()
        for ts in range(len(xchunks)):
            y = F.relu(self.conv1(xchunks[ts]))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            y = y.view(y.size(0), -1) # flattening
            #ycat = torch.cat((ycat, y), 0 )
        
            ycat = y.view(-1, 1, self.rnn_input_dim)   # Adding dimention 
            #print(ycat.size())
        #output_chunks = torch.chunk(ycat, self.unroll, 0)
        #for yt in range(len(output_chunks)):
            self.hidden = self.repackage_hidden(self.hidden)
            output, self.hidden = self.lstm(ycat, self.hidden)#output_chunks[yt], self.hidden)
            #data = [h.data for h in self.hidden]
            #print('Hiden', self.hidden)
            #print('output', output)
            #if ts is len(xchunks)-1:
                #break
            #else:
                #del output
        #output, self.hidden = self.lstm(output, self.hidden)
        #y = F.tanh(output)
        y = torch.squeeze(output,1)
        return y


class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x





