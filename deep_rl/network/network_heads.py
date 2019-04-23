#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
import math
import time
import sys

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi= self.body(tensor(x))
        #print(phi)
        #print(self.features_detached.shape)
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class ApnnNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body,apnn_conf_data):
        super(ApnnNet, self).__init__()
        #self.apnn_head_input = layer_init(nn.Linear(256, body.feature_dim))
        #for p in self.apnn_head_input.parameters():
    #        p.requires_grad=False
        self.apnn_conf_data=apnn_conf_data
        self.output_dim=output_dim
        self.apnn_head_output = layer_init(nn.Linear(body.feature_dim, output_dim))
        for p in self.apnn_head_output.parameters():
            p.requires_grad=False
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        # APNN parameters
        self.alpha=self.apnn_conf_data["alpha"]
        self.beta=self.apnn_conf_data["beta"]
        self.alpha_matrix=self.alpha*torch.ones(output_dim,body.feature_dim).cuda()
        self.beta_matrix=self.beta*torch.ones(output_dim,body.feature_dim).cuda()
        self.zeros_matrix=torch.zeros(output_dim,body.feature_dim).cuda()
        self.Theta=torch.zeros(output_dim,body.feature_dim).cuda()
        self.Eligbility_traces=torch.zeros(output_dim,body.feature_dim).cuda()
        self.modulation=self.apnn_conf_data["modulation"]
        self.tau_m=self.apnn_conf_data["tau_m"]
        self.tau_E=self.apnn_conf_data["tau_E"]
        self.a_learning_rate=self.apnn_conf_data["a_learning_rate"]
        self.baseline_modulation=self.apnn_conf_data["baseline_modulation"]
        self.step_previous=-1
        self.theta_low=self.apnn_conf_data["theta_low"]
        self.theta_high=self.apnn_conf_data["theta_high"]
        self.epsiode_trigger=self.apnn_conf_data["modulation_clip_change_episode_trigger"]
        self.no_top_features=int(round(self.apnn_conf_data["percent_of_top_features"]*body.feature_dim))
        self.no_bottom_features=int(round(self.apnn_conf_data["percent_of_bottom_features"]*body.feature_dim))
        self.step=0
        self.action_storage=[]
        self.to(Config.DEVICE)
        #normalize the weights
        with torch.no_grad():
            self.apnn_head_output.weight.div_(torch.norm(self.apnn_head_output.weight, dim=1, keepdim=True))

    def predict(self, x, steps, to_numpy=False):
        features = self.body(tensor(x))
        #if not isinstance(steps,np.ndarray):
        #    if steps==2:
        #        features_fixed=torch.zeros(1,16).cuda()
        #        features_fixed[0,2]=0.8
        #        features_fixed[0,12]=0.8
        #        features=features_fixed
        #    if steps==4:
        #        if self.action_storage==1:
        #            features_fixed=torch.zeros(1,16).cuda()
        #            features_fixed[0,11]=0.8
        #            features_fixed[0,15]=0.8
        #            features=features_fixed
        #        else:
        #            features_fixed=torch.zeros(1,16).cuda()
        #            features_fixed[0,0]=0.8
        #            features_fixed[0,5]=0.8
        #            features=features_fixed
            #else:
            #    features_fixed=torch.zeros(1,16).cuda()
            #    features_fixed[0,3]=0.5
            #    features_fixed[0,4]=0.8
            #    features=features_fixed
        self.y_apnn_in=features.detach()
        self.y_apnn_out=F.sigmoid(self.apnn_head_output(features)).detach()
        y_dqn_out=self.fc_head(features)
        self.y_dqn_out_detached=y_dqn_out.detach()
        #temporary ouptut choose q-values of maximum between the two
        #print(features.shape)
        y=y_dqn_out
        #y=self.y_apnn_out*y_dqn_out
        self.y_apnn_out.detach_()
        #compute hebbian rule theta, note here I assumed t_pt is zero (check later not sure this is correct)
        #y_detached_transposed=torch.transpose(y.detach(),0,1)
        #if to_numpy:
        #    ojas_multiplayer=self.y_apnn_in-(self.apnn_head_output.weight.data*y_detached_transposed)
        #    self.Theta=y_detached_transposed*ojas_multiplayer
        #    print(self.Theta.shape)
        #self.Theta=torch.mm(torch.transpose(y.detach(),0,1),self.y_apnn_in)
        #activation_multiplication_y_input=torch.transpose(y.detach()+self.y_apnn_out,0,1)
        #values, indices = activation_multiplication_y_input.max(0)
        #activation_multiplication_y_input[activation_multiplication_y_input!=values]=0
        #Activation_multiplication=torch.mm(activation_multiplication_y_input,self.y_apnn_in)
        #print(Activation_multiplication.shape)
        #Theta_alpha=torch.where(Activation_multiplication>self.theta_high,self.alpha_matrix,self.zeros_matrix)
        #Theta_beta=torch.where(Activation_multiplication<self.theta_low,self.beta_matrix,self.zeros_matrix)
        #self.Theta=Theta_alpha+Theta_beta
        #self.compute_eligbility_traces(1,0)
        #Theta=torch.where(Theta!=self.theta_low, self.zeros_matrix,Theta)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

    def compute_eligbility_traces_action(self,value,action):
        value_tensor=torch.from_numpy(value).cuda().unsqueeze(0)
        activation_multiplication_value=torch.transpose(value_tensor,0,1)
        mask=torch.ones(activation_multiplication_value.shape).byte()
        #print(action)
        mask[action]=0
        activation_multiplication_value[mask]=0
        #print(activation_multiplication_value)
        Activation_multiplication=torch.mm(activation_multiplication_value,self.y_apnn_in)
        delete,top_features_indecies=torch.topk(Activation_multiplication[action,:],self.no_top_features)
        delete,bottom_features_indecies=torch.topk(Activation_multiplication[action,:],self.no_bottom_features,largest=False)
        activation_mask=torch.zeros(Activation_multiplication.shape).cuda()
        activation_mask[action,top_features_indecies]=1
        activation_mask[action,bottom_features_indecies]=0
        Theta_alpha=torch.where(activation_mask>self.theta_high,self.alpha_matrix,self.zeros_matrix)
        Theta_beta=torch.where(activation_mask<self.theta_low,self.beta_matrix,self.zeros_matrix)
        Activation_multiplication[Activation_multiplication==0]=1
        self.Theta=(Theta_alpha+Theta_beta)*Activation_multiplication

        #print(self.Theta)
        self.compute_eligbility_traces(1,0)

    def compute_eligbility_traces(self,delta_step,index):
        if type(self.Eligbility_traces) is np.ndarray:
            self.Eligbility_traces[index]=self.Eligbility_traces[index]*math.exp(-delta_step/self.tau_E)+self.Theta[index]
        else:
            temp_elgibility_trace=self.Eligbility_traces
            self.Eligbility_traces=self.Eligbility_traces*math.exp(-delta_step/self.tau_E)+self.Theta
        #    difference_et=self.Eligbility_traces-temp_elgibility_trace
            #if ((difference_et>0)*(difference_et<0.1)).any():
            #    print((difference_et>0)*(difference_et<0.1))
        #        print(difference_et)
        #        print(temp_elgibility_trace)
        #        print(self.Eligbility_traces)
        #        print(self.Theta)
        #        sys.exit()

    def update(self,rewards,steps,episodes):
        index=0
        for episode in np.nditer(episodes):
            reward=np.asscalar(rewards[index])
            self.step=np.asscalar(steps[index])
            if episode>self.epsiode_trigger:
                self.baseline_modulation=self.apnn_conf_data["baseline_modulation_after_change"] #blue line value
                #note at the moment not sure what step should be
            #delta_step=self.step-self.step_previous[index]
            #print(self.step)
            #print(self.step_previous[index])
            #print(delta_step)
            #time.sleep(5)
            #self.modulation=self.modulation*math.exp(-delta_step/self.tau_m)+self.a_learning_rate*reward+self.baseline_modulation
            self.modulation=reward+self.baseline_modulation
            #print(delta_step)
            #if episode>self.epsiode_trigger:
            #    Eligbility_traces_max=self.apnn_conf_data["Eligbility_traces_clip_max_after_change"]
            #    Eligbility_traces_min=self.apnn_conf_data["Eligbility_traces_clip_min_after_change"]
            #    self.Eligbility_traces[index].clamp_(Eligbility_traces_min,Eligbility_traces_max)
            #else:
            #    Eligbility_traces_max=self.apnn_conf_data["Eligbility_traces_clip_max_before_change"]
            #    Eligbility_traces_min=self.apnn_conf_data["Eligbility_traces_clip_min_before_change"]
            #    self.Eligbility_traces[index].clamp_(Eligbility_traces_min,Eligbility_traces_max)
            weight_change=self.modulation*self.Eligbility_traces[index]
            #if episode>self.epsiode_trigger:
            #    weight_change_max=self.apnn_conf_data["Weight_clip_max_after_change"]
            #    weight_change_min=self.apnn_conf_data["Weight_clip_min_after_change"]
            #    weight_change.clamp_(weight_change_min,weight_change_max)
            #else:
            #    weight_change_max=self.apnn_conf_data["Weight_clip_max_before_change"]
            #    weight_change_min=self.apnn_conf_data["Weight_clip_min_before_change"]
            #    weight_change.clamp_(weight_change_min,weight_change_max)
                #print(weight_change)
            #self.apnn_head_output.weight.data=(self.apnn_head_output.weight.data+weight_change).clamp(-3,3)
            self.apnn_head_output.weight.data=(self.apnn_head_output.weight.data+weight_change)
            with torch.no_grad():
                self.apnn_head_output.weight.div_(torch.norm(self.apnn_head_output.weight, dim=1, keepdim=True))
            #print(self.apnn_head_output.weight)
            #time.sleep(2)
            self.step_previous[index]=self.step
            index=index+1

    def reset(self):
        self.step_previous=-1
        self.Eligbility_traces=torch.zeros(self.output_dim,self.body.feature_dim).cuda()
        self.modulation=0

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.reset_flag=False
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        self.body.reset_flag=self.reset_flag
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class ModDuelingNet(nn.Module, BaseNet):
    '''Yang implementation to take in two values of the state and the memory'''
    def __init__(self, action_dim, body):
        super(ModDuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, x_nm, to_numpy=False):
        phi = self.body(tensor(x),tensor(x_nm))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class CategoricalNetMod(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNetMod, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, x_mem, to_numpy=False):
        phi = self.body(tensor(x), tensor(x_mem))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class QuantileNetMod(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNetMod, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, x_mem, to_numpy=False):
        phi = self.body(tensor(x), tensor(x_mem))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, to_numpy=False):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v

class CategoricalActorCriticNet_L2M_Mod(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet_L2M_Mod, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, mem, action=None):
        obs = tensor(obs)
        mem = tensor(mem)
        phi = self.network.phi_body(obs, mem)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v
