import os, sys, cv2, random, numpy as np, collections, datetime
import torch, torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

sys.path.append("game/")
import wrapped_flappy_bird as game

# 超参数
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000  # 观察期步数
EXPLORE = 2000000  # epsilon 退火总步数
INITIAL_EPSILON = 0.1  # 原始代码 0.0001 基本不探索，改回 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
FRAME_PER_ACTION = 1
UPDATE_TIME = 100  # 目标网络同步步数
width = height = 80
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on:', device)


def preprocess(observation):
    obs = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, obs = cv2.threshold(obs, 1, 255, cv2.THRESH_BINARY)
    return obs[np.newaxis, :, :]  # shape (1,80,80)


class DeepNetWork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256), nn.ReLU())
        self.out = nn.Linear(256, ACTIONS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.out(x)


class BrainDQNMain:
    def __init__(self, actions):
        self.replayMemory = collections.deque(maxlen=REPLAY_MEMORY)
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        self.Q_net = DeepNetWork().to(device)
        self.Q_netT = DeepNetWork().to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=1e-6)

        # ---- 统计量 ----
        self.episode_count = 0
        self.episode_reward = 0
        self.episode_loss = 0
        self.episode_q = 0
        self.train_steps = 0

        self.ep_rewards = []  # 每个episode的总奖励
        self.ep_losses = []  # 每个episode的平均损失
        self.ep_qs = []  # 每个episode的平均Q值

        self.loss_history = []  # 每次训练的损失
        self.reward_history = []  # 每步的奖励

        self.load()

    def save(self):
        torch.save({'model': self.Q_net.state_dict(),
                    'opt': self.optimizer.state_dict()},
                   'checkpoint.pth')
        print('checkpoint saved.')

    def load(self):
        if os.path.exists('checkpoint.pth'):
            ckpt = torch.load('checkpoint.pth', map_location=device)
            self.Q_net.load_state_dict(ckpt['model'])
            self.Q_netT.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['opt'])
            print('checkpoint loaded.')

    def plot_metrics(self):
        os.makedirs('logs', exist_ok=True)
        stamp = datetime.datetime.now().strftime('%m%d_%H%M%S')

        plt.figure(figsize=(15, 5))

        # 绘制episode级别的指标
        plt.subplot(131)
        if self.ep_losses:
            plt.plot(self.ep_losses)
            plt.title('Episode Loss')
            plt.xlabel('Episode')
            plt.ylabel('Average Loss')

        plt.subplot(132)
        if self.ep_rewards:
            plt.plot(self.ep_rewards)
            plt.title('Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')

        plt.subplot(133)
        if self.ep_qs:
            plt.plot(self.ep_qs)
            plt.title('Episode Q-value')
            plt.xlabel('Episode')
            plt.ylabel('Average |Q|')

        plt.tight_layout()
        plt.savefig(f'logs/metrics_{stamp}.png')
        plt.close()

        # 绘制训练过程的详细指标
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        if self.loss_history:
            # 对损失进行平滑处理
            if len(self.loss_history) > 100:
                smooth_loss = np.convolve(self.loss_history, np.ones(100) / 100, mode='valid')
                plt.plot(smooth_loss)
                plt.title('Training Loss (Smoothed)')
            else:
                plt.plot(self.loss_history)
                plt.title('Training Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')

        plt.subplot(132)
        if self.reward_history:
            # 对奖励进行平滑处理
            if len(self.reward_history) > 100:
                smooth_reward = np.convolve(self.reward_history, np.ones(100) / 100, mode='valid')
                plt.plot(smooth_reward)
                plt.title('Step Reward (Smoothed)')
            else:
                plt.plot(self.reward_history)
                plt.title('Step Reward')
            plt.xlabel('Step')
            plt.ylabel('Reward')

        plt.subplot(133)
        if self.ep_rewards:
            # 显示最近100个episode的奖励
            recent_rewards = self.ep_rewards[-100:] if len(self.ep_rewards) > 100 else self.ep_rewards
            plt.plot(recent_rewards)
            plt.title('Recent Episode Rewards')
            plt.xlabel('Episode (relative)')
            plt.ylabel('Total Reward')

        plt.tight_layout()
        plt.savefig(f'logs/training_{stamp}.png')
        plt.close()

        print(f'saved logs/metrics_{stamp}.png and logs/training_{stamp}.png')

    def train(self):
        if len(self.replayMemory) < BATCH_SIZE:
            return 0

        batch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = torch.cat([torch.Tensor(b[0]).unsqueeze(0) for b in batch]).to(device)
        action_batch = torch.LongTensor([b[1] for b in batch]).to(device)
        reward_batch = torch.Tensor([b[2] for b in batch]).to(device)
        next_batch = torch.cat([torch.Tensor(b[3]).unsqueeze(0) for b in batch]).to(device)
        terminal_batch = torch.BoolTensor([b[4] for b in batch]).to(device)

        q_next = self.Q_netT(next_batch).max(1)[0].detach()
        y = reward_batch + (~terminal_batch) * GAMMA * q_next
        q_pred = self.Q_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 记录损失
        loss_value = loss.item()
        self.episode_loss += loss_value
        self.train_steps += 1
        self.loss_history.append(loss_value)

        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())

        # 保存网络
        if self.timeStep % 1000 == 0:
            self.save()

        return loss_value

    def getAction(self):
        state = torch.Tensor(self.currentState).unsqueeze(0).to(device)
        with torch.no_grad():
            qval = self.Q_net(state)[0]
        self.episode_q += qval.abs().mean().item()

        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                act_idx = random.randrange(self.actions)
            else:
                act_idx = qval.argmax().item()
            action[act_idx] = 1
        else:
            action[0] = 1  # do nothing

        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    def setPerception(self, nextObs, action, reward, terminal):
        self.episode_reward += reward
        self.reward_history.append(reward)

        newState = np.append(self.currentState[1:], nextObs, axis=0)
        self.replayMemory.append((self.currentState, action.argmax(),
                                  reward, newState, terminal))
        self.currentState = newState
        self.timeStep += 1

        if self.timeStep > OBSERVE:
            self.train()

        # ---- episode 结束，记录统计量 ----
        if terminal:
            self.episode_count += 1

            # 计算episode的平均值
            avg_loss = self.episode_loss / max(1, self.train_steps)
            avg_q = self.episode_q / self.timeStep  # 使用当前episode的总步数

            self.ep_losses.append(avg_loss)
            self.ep_rewards.append(self.episode_reward)
            self.ep_qs.append(avg_q)

            # 每10个episode打印一次统计信息
            if self.episode_count % 10 == 0:
                print(f'\nEpisode {self.episode_count}: Reward={self.episode_reward:.1f}, '
                      f'Avg Loss={avg_loss:.4f}, Avg Q={avg_q:.4f}, '
                      f'Epsilon={self.epsilon:.5f}')

            # 每100个episode保存一次图表
            if self.episode_count % 100 == 0:
                self.plot_metrics()

            # 重置episode统计量
            self.episode_reward = 0
            self.episode_loss = 0
            self.episode_q = 0
            self.train_steps = 0

        # ---- 打印进度 ----
        state = ''
        if self.timeStep <= OBSERVE:
            state = 'observe'
        elif self.timeStep <= OBSERVE + EXPLORE:
            state = 'explore'
        else:
            state = 'train'

        print(f'\rTimestep: {self.timeStep} | State: {state:8} | '
              f'Epsilon: {self.epsilon:.5f} | Episode: {self.episode_count} | '
              f'Current Reward: {reward:+.1f}', end='')

    def setInitState(self, obs):
        self.currentState = np.tile(obs, (4, 1, 1))  # (4,80,80)

# 主循环
if __name__ == '__main__':
    brain = BrainDQNMain(ACTIONS)
    game_state = game.GameState()

    # 初始动作：什么都不做
    do_nothing = np.array([1, 0])
    obs0, _, _ = game_state.frame_step(do_nothing)
    obs0 = preprocess(obs0)
    brain.setInitState(obs0)

    while True:
        act = brain.getAction()
        obs, reward, terminal = game_state.frame_step(act)
        obs = preprocess(obs)
        brain.setPerception(obs, act, reward, terminal)
