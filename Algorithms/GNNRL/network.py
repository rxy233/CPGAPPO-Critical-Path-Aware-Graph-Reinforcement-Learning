import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv
from collections import deque
from torch.nn import Linear
from torch_geometric.data import Data
from tensorboardX import SummaryWriter

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return (x for x in zip(*transitions))

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        # input_dim就是每个节点的state的长度
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 第一层GCN
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # 第二层GCN
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 第二层GCN
        self.fc = Linear(hidden_dim, output_dim)
        self.tanh = torch.tanh
        self.dropout = torch.nn.Dropout(p=0.05)

        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        # 第一层GCN
        x = self.conv1(x, edge_index)
        # x = self.tanh(x)
        x = F.relu(x)
        # 第二层GCN
        x = self.conv2(x, edge_index)
        # x = self.tanh(x)
        x = F.relu(x)

        # x = self.conv3(x, edge_index)
        # # x = self.tanh(x)
        # x = F.relu(x)

        # 池化
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        x = global_add_pool(x, None)
        # x = self.pool(x, None)
        return x


class Net(nn.Module):
    def __init__(self, basegraph_num, state_dim, action_dim=3, hidden_dim=128, output_dim=32, num_layer=3):
        # state_dim = 每个节点的特征维度，也是GNN的输入维度
        super(Net, self).__init__()
        self.basegraph_num = basegraph_num
        # 设置设备
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # SimpleGNN的output_dim是output_dim，所以GNN输出维度是output_dim
        self.gnn = SimpleGNN(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.state_dim = state_dim  # State dimension
        
        # GNN输出的是图级嵌入，维度为output_dim，直接构建从output_dim到action_dim的网络
        self.layers = nn.ModuleList()
        current_dim = output_dim  # GNN输出维度固定为output_dim
        
        for i in range(num_layer):
            if i == num_layer - 1:
                # 最后一层输出action_dim
                self.layers.append(nn.Linear(current_dim, action_dim))
            else:
                # 隐藏层，可以扩展到hidden_dim
                self.layers.append(nn.Linear(current_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                current_dim = hidden_dim

        # 初始化权重
        for m in self.layers:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, state):
        # GNN处理图数据，输出图级嵌入向量，维度为32
        embedding = self.gnn(state)  # 形状: [batch_size, 32] 或 [32]
        
        # 确保embedding是2D张量 [batch_size, 32]
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)  # 添加batch维度
            squeeze_output = True
        else:
            squeeze_output = False
            
        # 通过全连接层
        x = embedding
        for layer in self.layers:
            x = layer(x)
        
        # 如果输入是1D，输出也应该是1D
        if squeeze_output:
            x = x.squeeze(0)
            
        x = F.softmax(x, dim=-1)  # 在最后一个维度上做softmax
        return x



class DQN:
    def __init__(self, args, basegraph_num, state_dim, out_dim):
        # 设置设备
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # 获取隐藏层维度，如果args没有则默认128
        hidden_dim = getattr(args, 'hidden_dim', 128)
        # GNN输出维度固定为32
        gnn_output_dim = 32
        self._behavior_net = Net(basegraph_num, state_dim=state_dim, action_dim=out_dim, hidden_dim=hidden_dim, output_dim=gnn_output_dim).to(self.device)
        self._target_net = Net(basegraph_num, state_dim=state_dim, action_dim=out_dim, hidden_dim=hidden_dim, output_dim=gnn_output_dim).to(self.device)

        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        # self._optimizer = torch.optim.SGD(self._behavior_net.parameters(), lr=args.lr)
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr)

        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        self.args = args
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

        self.loss = 0

        self.writer = SummaryWriter(log_dir='./logs/dqn_logs')

    def select_action(self, state, epsilon, action_space):
        random_number = random.random()
        if random_number < epsilon:
            action = action_space.sample()
            # action = random.randint(0, 2)
        else:
            # 确保状态数据与模型在同一设备上
            if hasattr(state, 'to'):
                state = state.to(self.device)
                # 如果是PyG Data对象，确保所有内部张量都在正确设备上
                if hasattr(state, 'x'):
                    state.x = state.x.to(self.device)
                if hasattr(state, 'edge_index'):
                    state.edge_index = state.edge_index.to(self.device)
                if hasattr(state, 'batch') and state.batch is not None:
                    state.batch = state.batch.to(self.device)
            else:
                # 如果是普通tensor，直接移动到设备
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            actions_prob = self._behavior_net(state)
            action = torch.argmax(actions_prob)
            action = action.item()
        return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward], next_state,
                            [int(done)])

    def update(self, total_steps):

        if total_steps % self.freq == 0:
            # print("update -- freq")
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            print("update -- target_freq")
            self._update_target_network()
            # self.writer.add_scalar('Loss', self.loss, total_steps)

    def _update_behavior_network(self, gamma):
        losses = []
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._memory.sample(self.batch_size,
                                                                                                    )

        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch,
                                                           done_batch):

            state, action, reward, next_state, done = state, action[0], reward[0], next_state, done[0]
            # self.writer.add_graph(self._behavior_net.model, state)
            q_value = self._behavior_net(state)
            q_value = q_value[:, action]

            with torch.no_grad():
                if self.args.double:
                    # print("Double DQN")
                    _, a_next = torch.max(self._behavior_net(next_state), dim=1, keepdim=False)
                    q_next = self._target_net(next_state)[:, a_next][0]
                else:
                    q_next, _ = torch.max(self._target_net(next_state), dim=1, keepdim=False)
                q_target = reward + gamma * q_next  # * (1 - done)
                q_target = q_target.detach()
            criterion = nn.MSELoss()
            # print(f"\tq_target: {q_target}\tq_value: {q_value}")
            loss = criterion(q_target, q_value)
            losses.append(loss)

        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)

        self._optimizer.zero_grad()

        total_loss = torch.stack(losses).sum()
        total_loss.backward()

        self.loss = total_loss.item()

        self._optimizer.step()



    def _update_target_network(self):
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=True):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=True):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

    def finished(self):
        self.writer.close()
# # 创建GNN模型
# gnn = SimpleGNN(input_dim=3, hidden_dim=64, output_dim=32)
#
# # 前向传播
# embedding = gnn(graph_data)
