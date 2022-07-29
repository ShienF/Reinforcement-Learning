import socket
import json

action = {'title': 'action', 'content': {'voltage': []}}

class Server():
    def __init__(self, host = '127.0.0.1', port = 5006, listener = 5):
        self.host = host
        self.port = port
        self.lisetener = listener
        self.bufferSize = 12000
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # family:server to server; type:TCP
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((host, self.port))
        # self.s.setblocking(False) # non blocking
        self.s.listen(self.lisetener) # at most how many sockets connect
        print('server start at: %s:%s' % (self.host, self.port))
        print('wait for connection...')

        self.client, addr = self.s.accept() # wait for client request
        print('connected by' + str(addr))

        # self.sendAction({"voltage":[1,0,0,0,100,200,100,100]})

    def recvData(self): #while true??
        indata = self.client.recv(self.bufferSize).decode()
        indata = json.loads(indata)
        # print('Receive from Unity: ', indata)
        return indata

    def sendAction(self, action):
        self.client.send(json.dumps(action, indent = 4).encode())
        # print('send action to Unity...', action)

import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import os

class CriticNetwork(nn.Module):
    def __init__(self, q_lr, input_dims, fc1_dims, fc2_dims, n_actions, chept_dir, name):
        super(CriticNetwork, self).__init__()

        self.l1 = nn.Linear(input_dims +  n_actions, fc1_dims)
        self.l2 = nn.Linear(fc1_dims,fc2_dims)
        self.l3 = nn.Linear(fc2_dims, 1)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
    #     self.input_dims = input_dims
    #     self.fc1_dims = fc1_dims
    #     self.fc2_dims = fc2_dims
    #     self.n_actions = n_actions
        self.name = name
        self.chept_dir = chept_dir

    #     self.state_value = nn.Sequential(
    #         nn.Linear(self.input_dims, self.fc1_dims), # unpack tuple??
    #         nn.LayerNorm(self.fc1_dims), #bn
    #         nn.ReLU(),
    #         nn.Linear(self.fc1_dims, self.fc2_dims),
    #         nn.LayerNorm(self.fc2_dims)
    #     )

    #     self.action_value = nn.Sequential(
    #         nn.Linear(self.n_actions, self.fc2_dims),
    #         nn.ReLU()

    #     )

    #     self.q = nn.Sequential(
    #         nn.Linear(self.fc2_dims,1)
    #     )

        self.optimizer = optim.Adam(self.parameters(), lr = q_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.l1(torch.cat([state, action], 1))
        x = F.relu(x)
        x = self.bn1(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.l3(x)
        return x

    #     state_value = self.state_value(state)
    #     action_value = self.action_value(action)
    #     state_action_value = F.relu(torch.add(state_value, action_value))
    #     state_action_value = self.q(state_action_value)
        # return state_action_value
    
    def save_checkpoint(self, tag):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), os.path.join(self.chept_dir, self.name + '{t}' + '_ddpg').format(t = tag))
    
    def load_checkpoint(self, tag):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(os.path.join(self.chept_dir, self.name + '{t}' + '_ddpg').format(t = tag)))

class ActorNetwork(nn.Module):
    def __init__(self, pi_lr, input_dims, fc1_dims, fc2_dims, n_actions, chept_dir, name):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(input_dims, fc1_dims)
        self.l2 = nn.Linear(fc1_dims, fc2_dims)
        self.l3 = nn.Linear(fc2_dims, n_actions)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        # self.input_dims = input_dims
        # self.fc1_dims = fc1_dims
        # self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chept_dir = chept_dir
        

        # self.mu = nn.Sequential(
        #     nn.Linear(self.input_dims, self.fc1_dims),
        #     nn.LayerNorm(self.fc1_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.fc1_dims, self.fc2_dims),
        #     nn.LayerNorm(self.fc2_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.fc2_dims, self.n_actions)
        # )

        self.optimizer = optim.Adam(self.parameters(), lr = pi_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = torch.tanh(self.l3(x))
        return x

        # actions = self.mu(state)
        # actions = torch.tanh(actions)
        # return actions
    
    def save_checkpoint(self, tag):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), os.path.join(self.chept_dir, self.name + '{t}' + '_ddpg').format(t = tag))
    
    def load_checkpoint(self, tag):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(os.path.join(self.chept_dir, self.name + '{t}' + '_ddpg').format(t = tag)))

import numpy as np

class ReplayBuffer():
    def __init__(self, input_dims, n_actions, max_size = 5000): ### 10000
        self.size = max_size
        self.cntr = 0
        self.state_mem = np.zeros((self.size, input_dims))
        self.action_mem = np.zeros((self.size, n_actions))
        self.reward_mem = np.zeros(self.size) 
        self.new_state_mem = np.zeros((self.size, input_dims))
        self.terminal_mem = np.zeros(self.size, dtype = np.float32)
    
    def store_transition(self, s, a, r, s_, d):
        index = self.cntr % self.size
        self.state_mem[index] = s
        self.action_mem[index] = a
        self.reward_mem[index] = r
        self.new_state_mem[index] = s_
        self.terminal_mem[index] = d
        self.cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.cntr, self.size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminals

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

def four2eight(action):
    action_cpy = action.copy()
    # if init:
    #     for i in range(4):
    #         action_cpy.insert(0, 0) #根據target重訂 先固定轉向四軸 ###
    #     action_cpy[4] = s[1]
    #     action_cpy[5] = s[2]
    #     action_cpy[6] = 2000
    #     action_cpy[7] = 0

    #     return action_cpy

    for i in range(4):
        action_cpy.insert(0, 0)

    return action_cpy

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

class Agent():
    def __init__(self, q_lr, pi_lr, gamma, rho, server, input_dims = 19, n_actions = 4, ###
                layer1_size = 512, layer2_size = 512, batch_size = 128, chpt_dir = 'tmp/ddpg/0724'):
        self.rho = rho
        self.gamma = gamma
        self.batch_size = batch_size
        self.server = server
        self.memory = ReplayBuffer(input_dims = input_dims, n_actions = n_actions)
        self.noice = OUActionNoise(mu = np.zeros(n_actions))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(chpt_dir):
            os.mkdir(chpt_dir)

        self.critic = CriticNetwork(q_lr, input_dims, layer1_size, layer2_size, n_actions, chpt_dir,
                                    name = 'Crirtic_')
                
        self.actor = ActorNetwork(pi_lr, input_dims, layer1_size, layer2_size, n_actions, chpt_dir,
                                    name = 'Actor_')
        
        self.target_critic = CriticNetwork(q_lr, input_dims, layer1_size, layer2_size, n_actions, chpt_dir,
                                            name = 'TargetCrirtic_')
        
        self.target_actor = ActorNetwork(pi_lr, input_dims, layer1_size, layer2_size, n_actions, chpt_dir,
                                        name = 'TargetActor_')
        
        self.update_network_parameters(rho = 1)

    def update_network_parameters(self, rho = None):
        if rho is None:
            rho = self.rho

        critic_params = self.critic.named_parameters()
        actor_params = self.actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_params_dict = dict(critic_params)
        actor_params_dict = dict(actor_params)
        target_critic_params_dict = dict(target_critic_params)
        target_actor_params_dict = dict(target_actor_params)

        for name in critic_params_dict:
            critic_params_dict[name] = rho * critic_params_dict[name].clone() + \
                                    (1 - rho) * target_critic_params_dict[name].clone()
        self.target_critic.load_state_dict(critic_params_dict)
        
        for name in actor_params_dict:
            actor_params_dict[name] = rho * actor_params_dict[name].clone() + \
                                    (1 - rho) * target_actor_params_dict[name].clone()
        self.target_actor.load_state_dict(actor_params_dict)
    
    def choose_actions(self, observation, init = 0):
        if init:
            # actions = [100,100,100,100,1000,1000,500,500] #根據target重訂 先固定轉向四軸 ###
            mu = 0
            sigma = 1500
            r = np.random.normal(mu, sigma, 4)
            actions = np.clip(r, -2000, 2000)
            actions = actions.tolist()
            actions_ = four2eight(actions)
            actions_msg = {'title': 'action', 'content': {'voltage': actions_}}
            self.server.sendAction(actions_msg)
            return actions
            
        self.actor.eval()
        observation = torch.tensor(observation, dtype = torch.float).to(self.device)
        actions = self.actor.forward(observation).to(self.device)
        actions_ = actions + torch.tensor(self.noice(), dtype = torch.float).to(self.device)
        actions_ = actions_ * 2000
        actions_ = torch.clamp(actions_, -2000, 2000)
        actions_ = list(actions_.cpu().detach().numpy().tolist())
        actions__ = four2eight(actions_)
        self.server.sendAction({'title': 'action', 'content': {'voltage': actions__}})
        self.actor.train()
        # return actions_.cpu().detach().numpy()
        return actions_
    
    def learn(self):
        state, actions, reward, new_state, d = self.memory.sample_buffer(self.batch_size)
        state = torch.tensor(state, dtype = torch.float).to(self.device)
        actions = torch.tensor(actions, dtype = torch.float).to(self.device)
        reward = torch.tensor(reward, dtype = torch.float).to(self.device)
        new_state = torch.tensor(new_state, dtype = torch.float).to(self.device)
        d = torch.tensor(d).to(self.device)

        self.target_actor.eval()
        target_actions = self.target_actor.forward(new_state)
        self.target_critic.eval()
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        self.critic.eval()
        critic_value = self.critic.forward(state, actions)

        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + (1 - d[i]) * self.gamma * critic_value_[i])
        target = torch.tensor(target, dtype = torch.float).to(self.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad() #clean the previous grdient
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward() #claculate gradient
        self.critic.optimizer.step() #update paramters

        actions = self.actor.forward(state)
        self.actor.train()
        self.actor.optimizer.zero_grad()
        self.critic.eval()
        actor_loss = -self.critic.forward(state, actions)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def save_models(self, tag):
        self.critic.save_checkpoint(tag)
        self.actor.save_checkpoint(tag)
        self.target_critic.save_checkpoint(tag)
        self.target_actor.save_checkpoint(tag)
    
    def load_models(self, tag):
        self.critic.load_checkpoint(tag)
        self.actor.load_checkpoint(tag)
        self.target_critic.load_checkpoint(tag)
        self.target_actor.load_checkpoint(tag)

import torch
import numpy as np

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list(list_of_lists)
    if hasattr(list_of_lists[0], '__iter__'):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list(list_of_lists[:1]) + flatten(list_of_lists[1:])

def decomposeCosSin(angle):
    return [np.cos(angle), np.sin(angle)]

def processFeature(state:dict, targetPos):
    feature = []
    feature.append(state['baseLinkPos']['x']-targetPos[0])
    feature.append(state['baseLinkPos']['y']-targetPos[1])
    feature.append(decomposeCosSin(state['baseLinkOrientation']))
    feature.append(state['baseLinkVelocity']['x'])
    feature.append(state['baseLinkVelocity']['y'])
    feature.append(state['baseLinkAngularVelocity'])
    feature.append(decomposeCosSin(state['wheelBaseOrientation']))
    feature.append(state['wheelVelocity'])
    feature = flatten(feature)
    return feature

from threading import Thread
import math

class CustomThread(Thread):
    def __init__(self, server):
        Thread.__init__(self)
        self.server = server
        self.message = None
    
    def run(self):
        self.message = self.server.recvData()

class Environment():
    def __init__(self, server, target):
        self.server = server
        self.devie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pos = None
        self.target_pos = None
        self.target_real_pos = target
    
    def calculate_reward(seldf, pos, target_pos):
        reward = -math.dist(pos, target_pos)
        return reward

    def check_termination(self, pos, target_pos):
        distance = math.dist(pos, target_pos)
        return distance < 1.5 or distance > 10 ###
    
    def get_pos_dict(self, state):
        return [state['baseLinkPos']['x'], state['baseLinkPos']['y']]
    
    def get_pos(self, state):
        return [state[0], state[1]]
    
    def restart_episode(self):
        self.pos = [0., 0.]
        self.target_pos = [self.pos[0] + self.target_real_pos[0], self.pos[1] + self.target_real_pos[1]]
        new_target = {'title': 'new target', 'content': {'pos':{'x':self.target_pos[0],'y':0, 'z':self.target_pos[1]}}}
        self.server.sendAction(new_target)
        
    def step(self, obs):
        reward = None
        new_state = None
        t = CustomThread(self.server)
        t.start()
        t.join()
        
        if type(obs) == dict:
            prev_pos = self.get_pos_dict(obs)
        else:
            prev_pos = self.get_pos(obs)
        
        if t.message:
            self.pos = self.get_pos_dict(t.message)
            new_state = processFeature(t.message, self.target_pos)
            reward = self.calculate_reward(self.pos, self.target_pos) - self.calculate_reward(prev_pos, self.target_pos)
            done = self.check_termination(self.pos, self.target_pos)
            
        return reward, new_state, done

import matplotlib.pyplot as plt

def plot(reward, crtirc_loss, actor_loss, name, path, show = False):
    length = len(reward)

    x = [i for i in range(length)]
    
    figure, axis = plt.subplots(2,2)

    axis[0,0].plot(x, reward)
    axis[0,0].set_title('Reward')
    axis[1,0].plot(x, crtirc_loss)
    axis[1,0].set_title('Crtic_loss')
    axis[1,1].plot(x, actor_loss)
    axis[1,1].set_title('Actor_loss')
   
    axis[1,0].set(xlabel='epoch')
    axis[1,1].set(xlabel='epoch')
    axis[0,0].label_outer()
    axis[0,1].label_outer()

    plt.savefig(os.path.join(path + 'Training_records_at_{n}.png'.format(n = name)))
    if show:
        plt.show()

def main(mode):
    print('The mode is:',mode)
    s = Server()
    target = [-5.,5.]
    env = Environment(server = s, target = target)
    chpt_dir_ = 'tmp/ddpg/0727'
    agent =  Agent(q_lr = 0.001, pi_lr = 0.0001, gamma = 0.99, rho = 0.995, server = s, chpt_dir = chpt_dir_)
    epoch = 10000

    reward_history = []

    if mode == 'train':
        critic_loss_history = []
        actor_loss_history = []

        load_epoch = 700 # 0
        agent.load_models(load_epoch)
        for i in range(epoch):
            env.restart_episode()
            t = CustomThread(s)
            t.start()
            t.join()
            obs = t.message #dict

            done = False
            score = 0
            c_loss = 0
            a_loss = 0
            init = 1
            
            ctr = 0
            while(not done and ctr < 200):
                ctr += 1
                # print(ctr, '\n')
                if init:
                    actions = agent.choose_actions(obs, 1)
                    init = 0
                else: 
                    actions = agent.choose_actions(obs)
                
                reward, new_state, done = env.step(obs)
                # reward -= 10
                if type(obs) == dict:
                    agent.memory.store_transition(processFeature(obs, target), actions, reward, new_state, int(done))
                else:
                    agent.memory.store_transition(obs, actions, reward, new_state, int(done))
                
                critic_loss, actor_loss = agent.learn()

                score += reward
                c_loss += critic_loss
                a_loss += actor_loss

                obs = new_state.copy()

            reward_history.append(score)
            critic_loss_history.append(c_loss)
            actor_loss_history.append(a_loss)
        
            if (i+1) % 100 == 0:
                agent.save_models(load_epoch+i+1)
                plot(reward_history, critic_loss_history, actor_loss_history, load_epoch+i+1, path = chpt_dir_)
            
            print('episode: ', load_epoch+i, 
            'reward: %.2f'%(score),
            'training 100 times avg reward: %.3f'%(np.mean(reward_history[-100:])),
            'critic loss: %.2f'%(c_loss),
            'actor loss: %.2f'%(a_loss))
        
        plot(reward_history, critic_loss_history, actor_loss_history, load_epoch+epoch , show = True, path = chpt_dir_)
    
    elif mode == 'test':
        agent.load_models(6000) #1950/2800/2900/4900/5450/6150/7150/8500
        for i in range(10):
            
            env.restart_episode()

            t = CustomThread(s)
            t.start()
            t.join()
            obs = t.message
            obs = processFeature(obs, target)

            done = False
            score = 0
            ctr = 0

            while(not done and ctr < 250):
                ctr += 1
                actions = agent.choose_actions(obs)
                reward, new_state, done = env.step(obs)
                score += reward
                obs = new_state.copy()
            
            print('test run: ', i, 
            'reward: %.2f'%(score))
    

if __name__ == '__main__':
    mode = 'train'
    # mode = 'test'
    main(mode)
