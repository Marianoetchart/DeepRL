#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *


class ApnnDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

    def episode(self, deterministic=False,episode=0):
        episode_start_time = time.time()
        #print 'begin reset'
        state = self.task.reset()[0]
        state=np.expand_dims(state,axis=0)
        total_reward = 0.0
        steps = 0
        self.network.reset()
        y_apnn_out_stack=np.zeros((self.network.output_dim))
        y_apnn_in_stack=np.zeros((self.network.body.feature_dim))
        y_dqn_out_stack=np.zeros((self.network.output_dim))
        y_conv_layers_stack=np.zeros((256))
        Eligbility_traces_stack=np.zeros((self.network.output_dim,self.network.body.feature_dim))
        action_list=[]
        #print('begin episode')
        #print(state)
        while True:
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]),steps, True).flatten()
            value=(value+self.network.y_apnn_out.cpu().detach().numpy()).flatten()
            #print(value)
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            #print 'call self.task.step(action)'
            next_state, reward, done, info = self.task.step(action)
            self.network.compute_eligbility_traces_action(value,action)
            action_list.append(action)
            if steps==2:
                action_storage=action
                self.network.action_storage=action_storage
            #action_list.append(action)
            y_apnn_out_stack=np.vstack([y_apnn_out_stack, self.network.y_apnn_out.cpu()])
            y_apnn_in_stack=np.vstack([y_apnn_in_stack, self.network.y_apnn_in.cpu()])
            y_dqn_out_stack=np.vstack([y_dqn_out_stack, self.network.y_dqn_out_detached.cpu()])
            Eligbility_traces_stack=np.dstack([Eligbility_traces_stack, self.network.Eligbility_traces.cpu()])
            y_conv_layers_stack=np.vstack([y_conv_layers_stack, self.network.body.conv_layers_output.cpu()])
            #if(episode % 100 == 0) and done:
            #print(info)
            #print(next_state)
            #print(done)
            if len(next_state.shape)<3:
                next_state=np.expand_dims(next_state,axis=0)
            #print 'task step'
            #if reward >0.5:
            #    print(info)
            #    print(reward)
            #    print(action_list)
             #   time.sleep(5)
            total_reward += reward
            #if total_reward==1:
            #    print(action_list)
        #        time.sleep(1)
            reward = self.config.reward_normalizer(reward)
            #self.network.compute_eligbility_traces(1,0)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done),\
                 self.network.Eligbility_traces, self.network.modulation,\
                  self.network.step, self.network.step_previous,episode, self.network.Theta])
                self.total_steps += 1
            #self.network.update(reward,steps,episode)
            steps += 1
            state = next_state
            #print 'learning'
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, Eligbility_traces, modulation, network_step, network_step_previous,episode_replay, thetas = experiences
                #print(type(modulation))
                #print(Eligbility_traces[0])
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                #self.target_network.Eligbility_traces=Eligbility_traces
                #self.target_network.modulation=modulation
                #self.target_network.step_previous=network_step_previous
                #self.target_network.Theta=thetas
                #self.target_network.update(rewards,network_step,episode_replay)
                q_next = self.target_network.predict(next_states, network_step, False).detach()
                if self.config.double_q:
                    _, best_actions = self.network.predict(next_states).detach().max(1)
                    q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next, _ = q_next.max(1)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).unsqueeze(1).long()
                Eligbility_traces_temp=self.network.Eligbility_traces
                modulation_temp=self.network.modulation
                network_step_previous_temp=self.network.step_previous
                theta_temp=self.network.Theta
                self.network.Eligbility_traces=Eligbility_traces
                self.network.modulation=modulation
                self.network.step_previous=network_step_previous
                self.network.Theta=thetas
                self.network.update(rewards,network_step,episode_replay)
                q = self.network.predict(states,network_step, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                self.network.Eligbility_traces=Eligbility_traces_temp
                self.network.modulation=modulation_temp
                self.network.step_previous=network_step_previous_temp
                self.network.Theta=theta_temp
            #print 'self evaluate'
            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
            #print 'chekc is done'
            if done:
                #print 'end eposide'
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        if(episode % 1000 == 0):
            file_path=self.config.log_dir+"/debug_matrix/"
            np.save(file_path+"y_apnn_out_stack_{0}".format(episode),y_apnn_out_stack)
            np.save(file_path+"y_dqn_out_stack_{0}".format(episode),y_dqn_out_stack)
            np.save(file_path+"y_apnn_in_stack_{0}".format(episode),y_apnn_in_stack)
            np.save(file_path+"y_conv_layers_stack_{0}".format(episode),y_conv_layers_stack)
            np.save(file_path+"Eligbility_traces_stack_{0}".format(episode),Eligbility_traces_stack)
            np.save(file_path+"apnn_weights_{0}".format(episode),self.network.apnn_head_output.weight.data.cpu().detach())
            np.save(file_path+"reward_{0}".format(episode),total_reward)
        return total_reward, steps

#********************* BUGGY DO NOT USE****************************************
class ApnnQRDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state):
        value = self.network.predict(np.stack([self.config.state_normalizer(state)])).squeeze(0).detach()
        value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
        return np.argmax(value)

    def episode(self, deterministic=False,episode=0):
        self.target_network.zero_grad()
        self.network.zero_grad()
        episode_start_time = time.time()
        state = self.task.reset()[0]
        state=np.expand_dims(state,axis=0)
        total_reward = 0.0
        steps = 0
        action_list=[]
        while True:
            prediction_tensor=np.stack([self.config.state_normalizer(state)])
            value = self.network.predict(prediction_tensor).squeeze(0).detach()
            value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, info = self.task.step(action)
            action_list.append(action)
            if len(next_state.shape)<3:
                next_state=np.expand_dims(next_state,axis=0)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            #if episode>240000 and (action==1 or action==0):
            #    print(value)
            #    print(action_list)
            #    print(reward)
            #    time.sleep(2)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done), self.network.Eligbility_traces, episode])
                self.total_steps += 1
            self.network.update(reward,steps,episode)
            steps += 1
            state = next_state

            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, Eligbility_traces = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                self.target_network.Eligbility_traces= Eligbility_traces
                quantiles_next = self.target_network.predict(next_states).detach()
                q_next = (quantiles_next * self.quantile_weight).sum(-1)
                print(q_next)
                print(torch.max(q_next, dim=1))
                _, a_next = torch.max(q_next, dim=1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, quantiles_next.size(2))
                quantiles_next = quantiles_next.gather(1, a_next).squeeze(1)

                rewards = tensor(rewards)
                terminals = tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next
                self.network.Eligbility_traces=Eligbility_traces
                quantiles = self.network.predict(states)
                actions = tensor(actions).long()
                actions = actions.view(-1, 1, 1).expand(-1, -1, quantiles.size(2))
                quantiles = quantiles.gather(1, actions).squeeze(1)

                quantiles_next = quantiles_next.t().unsqueeze(-1)
                diff = quantiles_next - quantiles
                loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()

                self.optimizer.zero_grad()
                loss.mean(0).mean(1).sum().backward()
                self.optimizer.step()
                self.target_network.body.reset_flag=True

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
