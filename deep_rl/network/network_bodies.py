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
        print("First:",x.size())
        y = F.relu(self.conv1(x))
        print("Second:",y.size())
        y = F.relu(self.conv2(y))
        print("Third:",y.size())
        y = F.relu(self.conv3(y))
        print("Fourth:",y.size())
        y = y.view(y.size(0), -1)
        print("Fifth:",y.size())
        y = F.relu(self.fc4(y))
        print("Sixth:",y.size())
        return y

class DRQNBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DRQNBody, self).__init__()
        self.feature_dim = 512
        self.rnn_input_dim = 7*7*64
        self.batch_size = 1
        self.unroll  = 4
        in_channels = 1 # for 1 frame input
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.lstm = nn.LSTM(self.rnn_input_dim, self.feature_dim , num_layers = 1)
        self.hidden = self.init_hidden()
        self.reset_flag = False

    def init_hidden(self, num_layers = 1, batch = 1): 
        # initializing the hidden and cell states
        if str(Config.DEVICE) == 'cpu':
            return (autograd.Variable(torch.zeros(num_layers, batch,self.feature_dim)),
                    autograd.Variable(torch.zeros(num_layers, batch, self.feature_dim)))
        else: 
            return (autograd.Variable(torch.zeros(num_layers, batch,self.feature_dim)).cuda(),
                    autograd.Variable(torch.zeros(num_layers, batch, self.feature_dim)).cuda())

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(state) for state in h)

    def forward(self, x):
        if self.reset_flag:
            self.hidden = self.repackage_hidden(self.hidden)
            self.reset_flag = False
        
        batch = x.size(0)
        output = torch.Tensor()
        ycat = torch.Tensor()
        
        xchunks= torch.chunk(x,self.unroll, 1)
        self.hidden = self.init_hidden(batch = batch)

        for ts in range(len(xchunks)):
            y = F.relu(self.conv1(xchunks[ts]))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            y = y.view(batch, -1) # flattening
            #ycat = torch.cat((ycat, y), 0 )
        
            ycat = y.view(-1, batch, self.rnn_input_dim)   # Adding dimention 
            #print(ycat.size())
        #output_chunks = torch.chunk(ycat, self.unroll, 0)
        #for yt in range(len(output_chunks)):
            self.lstm.flatten_parameters()
            output, self.hidden = self.lstm(ycat, self.hidden)#output_chunks[yt], self.hidden)
            #data = [h.data for h in self.hidden]
            #print('Hiden', self.hidden)
            #print('output', output)
            #if ts is len(xchunks)-1:
                #break
            #else:
                #del output
        
        y = output.view(batch, -1)
        return y


class SpatialAttDRQNBody(nn.Module):
    def __init__(self, in_channels=4):
        super(SpatialAttDRQNBody, self).__init__()
        self.feature_dim = 256
        self.rnn_input_dim = 256
        self.batch_size = 1
        self.num_layers = 1
        self.unroll  = 4
        in_channels = 1 # for 1 column
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias= False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias = False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, bias  = False)
        self.att1 = nn.Linear(self.feature_dim , self.feature_dim, bias= False)
        self.att2 = nn.Linear(self.feature_dim, self.feature_dim, bias = False)
        self.lstm = nn.LSTM(self.rnn_input_dim, self.feature_dim, num_layers = self.num_layers)
        self.hidden = self.init_hidden()
        self.reset_flag = False

    def init_hidden(self, num_layers = 1, batch = 1): 
        # initializing the hidden and cell states
        if str(Config.DEVICE) == 'cpu':
            return (autograd.Variable(torch.zeros(num_layers, batch,self.feature_dim)),
                    autograd.Variable(torch.zeros(num_layers, batch, self.feature_dim)))
        else: 
            return (autograd.Variable(torch.zeros(num_layers, batch,self.feature_dim)).cuda(),
                    autograd.Variable(torch.zeros(num_layers, batch, self.feature_dim)).cuda())

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(state) for state in h)

    def forward(self, x):
        if self.reset_flag:
            self.hidden = self.repackage_hidden(self.hidden)
            self.reset_flag = False

        batch = x.size(0)
        output = torch.Tensor()
        
        xchunks= torch.chunk(x,self.unroll, 1)
        self.hidden = self.init_hidden(num_layers = self.num_layers, batch = batch)
        
        for ts in range(len(xchunks)):
            y = F.relu(self.conv1(xchunks[ts]))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y)) 
            y = y.view(batch, -1, self.feature_dim,).detach() # (batch) x 49 (input vector) x 256 (dimension)
            hidden= self.hidden[0].view(batch,-1, self.feature_dim).detach() # reshaping hidden state
            
            # Attention Network
            ht_1 = self.att1(hidden)
            del hidden
            xt_1 = self.att1(y)
            #print("ht_1:", ht_1.size())
            #print("xt_1:", xt_1.size())
            combined_att = ht_1.detach() + xt_1.detach()
            del ht_1, xt_1 
            #print(combined_att.requires_grad)
            combined_att = F.tanh(combined_att)
            #print("comb_att:", combined_att.size())
            combined_att2 = self.att2(combined_att.detach())

            #print("comb_att2:", combined_att2.size())
            goutput = F.softmax(combined_att2.detach(), dim=2)
            del combined_att2, combined_att
            #print("goutput:", goutput.size())
            goutput = goutput.view(goutput.size(0),self.feature_dim, -1)
            context = torch.bmm(goutput, y) # dot product 
            del goutput, y 
            context = context.view(-1, batch, self.rnn_input_dim)   # Adding dimension for lstm 
            #self.hidden = self.repackage_hidden(self.hidden) # repackage hidden
            self.lstm.flatten_parameters()
            output, self.hidden = self.lstm(context, self.hidden) #LSTM
            del context
        
        y = output.view(batch, -1) # flattens output
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





