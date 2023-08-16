import numpy as np
from scipy.stats import norm
from forest.benchmarking.distance_measures import total_variation_distance
from Receiver import Softmax
from scipy.stats import rankdata

def genCommitment(sDim, mDim, num):
    commitments = np.zeros((num, sDim, mDim + 1))
    commitments[:, :, -1] = 1
    commitments[:, :, 1 : -1] = np.random.rand(num, sDim, mDim-1)
    commitments = np.sort(commitments, axis = -1)
    commitments = commitments[:, :, 1:] - commitments[:, :, :-1]
    return commitments


def vectorized_sample(prior, commitment, size):
    states = np.random.choice(len(prior), size, p=prior)
    cum_commitment = np.cumsum(commitment, axis = 1)
    p = np.random.uniform(0, 1, size=(size, 1))
    messages = (p < cum_commitment[states]).argmax(axis=1).astype(int)
    return states, messages

def collect_for_training(commitment, collect_data, data, m_dim, a_dim):
    message_stat = set()
    probability = np.zeros((m_dim, a_dim))
    for i in data:
        action = i % a_dim
        message = i // m_dim
        message_stat.add(message)
        probability[message, action] += 1
    for i in range(m_dim):
        if(np.sum(probability[i]) != 0):
            probability[i] = probability[i] / np.sum(probability[i])
    for message in message_stat:
        tmp = commitment[:, message].tolist()
        tmp.extend(probability[message].tolist())
        collect_data.append(tmp)

def interact_with_receiver(tester, commitment, prior, receiver_utility, m_dim, a_dim, tau, total_nums, sample_size):
    receiver = Softmax(m_dim, a_dim, tau=tau)
    states, messages = vectorized_sample(prior, commitment, total_nums)
    data_collect = []
    data_states = []
    for i, (state, message) in enumerate(zip(states, messages)):
        actor = receiver.select_action(message)
        reward = receiver_utility[state, actor]
        receiver.update_weight(reward)
        data_collect.append(message * m_dim + actor)
        data_states.append(state)
        if((i + 1) % (2 * sample_size) == 0):
            if(tester.isConverge(data_collect, 0.1)):
                break
            data_collect = []
            data_states = []
    return data_states, data_collect

class EarlyStopper(object):
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if (validation_loss + self.min_delta < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif abs(validation_loss - self.min_validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class Tester(object):
    def __init__(self, m_dim, a_dim, sample_size, t):
        self.m_dim = m_dim
        self.a_dim = a_dim
        self.sample_size = sample_size
        self.stat_N = np.zeros(t)
        self.t = t
        self.cnt = 0 #计算此时的cnt数据

    def isConverge(self, sample_data, eps):
        ranks = rankdata(sample_data, method='average')
        sum_the_first_half = np.sum(ranks[:self.sample_size]) - (self.sample_size * (self.sample_size + 1)) / 2.0
        indicator_value = 1 if abs(sum_the_first_half - (self.sample_size**2)/2) < (self.sample_size**2*eps) else 0
        self.stat_N[self.cnt] = indicator_value
        self.cnt = (self.cnt + 1) % self.t
        return np.sum(self.stat_N) > (self.t / 2.0)