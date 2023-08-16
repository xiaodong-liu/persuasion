import numpy as np

# EXP3算法是针对adversary bandit，并不能针对stochastic bandit
class Exp3(object):

    def __init__(self, m_dim, a_dim, lr):
        super(Exp3, self).__init__()
        self.m_dim = m_dim
        self.a_dim = a_dim
        self.lr = lr
        self.weights = np.ones((m_dim, a_dim))

    def action(self, message):
        prob = self.weights[message]
        self.prob = prob / np.sum(prob)
        self.a = np.random.choice(self.a_dim, p = self.prob)
        return self.a

    def update_weights(self, message, rewards):
        loss = np.zeros(self.a_dim)
        loss[self.a] = rewards / self.prob[self.a]
        self.weights[message] = self.weights[message] * np.exp(-self.lr * loss)


# 经过实验，基本能收敛到最优的结果, 但是收敛的行为不是一个稳定的分布
class UCB(object):
    def __init__(self, m_dim, a_dim):
        super(UCB, self).__init__()
        self.m_dim = m_dim
        self.a_dim = a_dim
        self.mean_reward = np.zeros((self.m_dim, self.a_dim))
        self.is_exploration = np.zeros((self.m_dim, self.a_dim), dtype=int)
        self.arm_count = np.zeros((self.m_dim, self.a_dim))

    def select_action(self, message):
        self.message = message
        flag = True
        for i in range(len(self.is_exploration[self.message])):
            if(self.is_exploration[self.message][i] == 0):
                self.action = i
                self.is_exploration[self.message][i] = 1
                flag = False
                break
        # 计算weights并选择最大的一个作为action
        print(self.arm_count[self.message])
        if flag:
            weights = self.mean_reward[self.message] + np.sqrt(np.log(np.sum(self.arm_count[self.message])) / self.arm_count[self.message])
            self.action = np.argmax(weights)
        return self.action


    def update_weight(self, reward):
        self.mean_reward[self.message][self.action] = (self.mean_reward[self.message][self.action] * self.arm_count[self.message][self.action])+ reward
        self.arm_count[self.message][self.action] += 1
        self.mean_reward[self.message][self.action] /= self.arm_count[self.message][self.action]

class Softmax(object):
    def __init__(self, m_dim, a_dim, tau=1.0):
        super(Softmax, self).__init__()
        self.tau = tau
        self.m_dim = m_dim
        self.a_dim = a_dim
        self.q_table = np.zeros((m_dim, a_dim))
        self.count = np.zeros((m_dim, a_dim))
    def select_action(self, message):
        self.message = message
        weights = np.exp(self.q_table[self.message] / self.tau)
        # 归一化
        prob = weights / np.sum(weights)
        self.action = np.random.choice(self.a_dim, p = prob)
        return self.action

    def update_weight(self, reward):
        self.count[self.message, self.action] += 1
        self.q_table[self.message, self.action] = self.q_table[self.message, self.action] + (reward - self.q_table[self.message, self.action]) / self.count[self.message, self.action]
        
