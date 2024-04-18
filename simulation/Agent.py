import numpy as np


class MWUAgent():

    def __init__(self, value, mechanism, full_info, lr):
        self.value = value
        self.mechanism = mechanism
        self.full_info = full_info
        self.lr = lr
        self.rho = self.value
        self.range = self.value + 1

        # MWU weights & info
        self.weights = np.ones(self.range) / self.range
        self.last_action = None

    def __sample_bid(self, test=False):
        if test:
            return np.argmax(self.weights)
        else:
            return np.random.choice(np.arange(self.range), size=1, p=self.weights)[0]

    def generate_action(self, test=False):
        if test:
            action = self.__sample_bid(test=True)
        else:
            action = self.__sample_bid(test=False)
            self.last_action = action

        return action

    def update_policy(self, last_reward, market_price, market_num):
        if self.full_info:
            # full info (experts setting), generate all utility
            virtual_bid = np.arange(self.range)
            if self.mechanism == 'second_price':
                rewards = np.where(virtual_bid >= market_price,
                                   self.value - market_price, 0.0)
            else:
                rewards = np.where(virtual_bid >= market_price,
                                   self.value - virtual_bid, 0.0)
            if market_price < self.range:
                rewards[market_price] = rewards[market_price] / \
                    (1.0 + market_num)
            rho = max(np.abs(rewards).max(), 1)
            pos_rate = (1 + self.lr) ** (rewards/rho)
            neg_rate = (1 - self.lr) ** (rewards/rho)
            # pos_rate = (1 + self.lr) ** (rewards/self.rho)
            # neg_rate = (1 - self.lr) ** (rewards/self.rho)
            self.weights *= np.where(rewards > 0, pos_rate, neg_rate)
        else:
            # bandit setting, only action's utility
            last_bid = self.last_action
            if last_reward >= 0:
                self.weights[last_bid] *= (1 + self.lr) ** (last_reward/self.rho)
            else:
                self.weights[last_bid] *= (1 - self.lr) ** (last_reward/self.rho)

        # normalize
        self.weights /= self.weights.sum()

    def test_policy(self):
        return self.__sample_bid(test=True)
