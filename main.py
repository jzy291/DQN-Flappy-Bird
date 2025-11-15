import pdb
import cv2
import sys
import os

sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn

# 超参数区
GAME = 'bird'  # 日志文件名标识
ACTIONS = 2  # 动作空间维度（跳或不跳）
GAMMA = 0.99  # 贝尔曼方程折扣因子
OBSERVE = 1000.  # 训练前先观察（纯累积经验）的步数
EXPLORE = 2000000.  # epsilon 从初始值降到最终值所需的步数
FINAL_EPSILON = 0.0001  # 最小探索率
INITIAL_EPSILON = 0.0001  # 初始探索率（源码已设得很小，几乎不随机）
REPLAY_MEMORY = 50000  # 经验池最大容量
BATCH_SIZE = 32  # 每次训练采样的批大小
FRAME_PER_ACTION = 1  # 每多少帧才允许决策一次（这里=1表示每帧都决策）
UPDATE_TIME = 100  # 每隔多少步把主网络权重同步给目标网络
SAVE_MODEL_TIME = 1000  # 每隔多少步保存权重文件
width = 80  # 预处理后的画面宽
height = 80  # 预处理后的画面高


# 图像预处理
def preprocess(observation):
    # 先缩放到 80×80，再转灰度图
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 二值化：像素>1 的置 255，其余 0，去掉背景细节
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    # 返回形状 (1,80,80) 的 float32 数组，供后续使用
    return np.reshape(observation, (1, 80, 80))


# 网络结构
class DeepNetWork(nn.Module):
    def __init__(self, ):
        super(DeepNetWork, self).__init__()
        # 输入 4 帧 80×80，输出 2 个动作 Q 值
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # 全连接：1600→256→2
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.out(x)


class BrainDQNMain(object):
    def save(self):
        """把主网络权重落盘到 params.pth"""
        print("save model param")
        torch.save(self.Q_net.state_dict(), 'params.pth')

    def load(self):
        """若存在存档则把权重同时载入主网络与目标网络"""
        if os.path.exists("params.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('params.pth'))
            self.Q_netT.load_state_dict(torch.load('params.pth'))

    def __init__(self, actions):
        self.replayMemory = deque()  # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.Q_net = DeepNetWork()  # 主网络
        self.Q_netT = DeepNetWork()  # 目标网络（固定一段时间再更新）
        self.load()  # 尝试载入存档
        self.loss_func = nn.MSELoss()  # 回归损失
        LR = 1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    # 训练
    def train(self):
        # 从回放池中随机采样 BATCH_SIZE 条转移
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]  # 当前状态
        action_batch = [data[1] for data in minibatch]  # 动作（one-hot）
        reward_batch = [data[2] for data in minibatch]  # 立即奖励
        nextState_batch = [data[3] for data in minibatch]  # 下一状态

        # 计算目标 y（贝尔曼目标）
        y_batch = np.zeros([BATCH_SIZE, 1])
        nextState_batch = np.array(nextState_batch)
        nextState_batch = torch.Tensor(nextState_batch)
        action_batch = np.array(action_batch)
        index = action_batch.argmax(axis=1)  # 取出实际执行动作的索引
        print("action " + str(index))
        index = np.reshape(index, [BATCH_SIZE, 1])
        action_batch_tensor = torch.LongTensor(index)  # 转 LongTensor 供 gather 使用

        # 用目标网络计算下一状态所有动作的 Q 值
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch = QValue_batch.detach().numpy()

        # 根据是否终止，计算 y = r + gamma * max Q(s', a')
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])

        # 主网络预测 Q(s, a) 并提取实际动作对应的 Q 值
        state_batch_tensor = Variable(torch.Tensor(state_batch))
        y_batch_tensor = Variable(torch.Tensor(y_batch))
        y_predict = self.Q_net(state_batch_tensor).gather(1, action_batch_tensor)

        # 计算 MSE 损失并反向传播
        loss = self.loss_func(y_predict, y_batch_tensor)
        print("loss is " + str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每 UPDATE_TIME 步把主网络权重同步给目标网络
        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())

        # 每 SAVE_MODEL_TIME 保存权重文件
        if self.timeStep % SAVE_MODEL_TIME == 0:
            self.save()



    def setPerception(self, nextObservation, action, reward, terminal):
        """
        环境每步调用此接口：
        1. 把旧状态、动作、奖励、新状态、终止信号打包存入经验池
        2. 经验池满后丢弃最早样本
        3. 观察期结束后开始训练
        4. 打印当前阶段（observe / explore / train）与 epsilon
        """
        # 用最新一帧替换掉 4 帧中的最旧一帧
        newState = np.append(self.currentState[1:, :, :], nextObservation, axis=0)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:  # Train the network
            self.train()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        """
        epsilon-greedy 策略：
        若随机数 < epsilon 则随机选动作，否则选 Q 值最大的动作。
        随时间步线性降低 epsilon。
        """
        currentState = torch.Tensor([self.currentState])
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                print("choose random action " + str(action_index))
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue.detach().numpy())
                print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    def setInitState(self, observation):
        """
        把第一帧复制 4 份，堆成初始 4 帧状态。
        observation 形状需为 (1,80,80)
        """
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)
        print(self.currentState.shape)


if __name__ == '__main__':
    actions = 2
    brain = BrainDQNMain(actions)
    flappyBird = game.GameState()
    action0 = np.array([1, 0])
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)
    print(brain.currentState.shape)
    while 1 != 0:
        action = brain.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal)
