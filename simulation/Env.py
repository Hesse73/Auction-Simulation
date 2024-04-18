import numpy as np
from Mechanism import first_price_auction, second_price_auction
from Plot import *
from Agent import MWUAgent
from NE import calculate_NE, calculate_expl


class PrivateAuction():

    def __init__(self, args):
        # players
        self.player_num = args.player_num
        self.agt_values = args.agt_values
        # set agents
        self.agt_list = [MWUAgent(value=args.agt_values[i],
                                  mechanism=args.mechanism,
                                  full_info=args.full_info,
                                  lr=args.lr)
                         for i in range(self.player_num)]
        # reward info
        self.full_info = args.full_info
        # game rounds
        self.max_rounds = args.max_rounds
        # save args
        self.args = args
        # set mechanism function
        self.mechanism = args.mechanism
        if self.mechanism == 'first_price':
            self.auction_func = first_price_auction
        elif self.mechanism == 'second_price':
            self.auction_func = second_price_auction
        else:
            raise NotImplementedError(f"Unknown mechanism: {self.mechanism}")
        # results saving
        self.estimate_freq = args.estimate_freq
        self.log_freq = args.log_freq
        # theoretical result
        self.theortc_utilitys, self.theortc_revenue = calculate_NE(
            self.agt_values, self.mechanism)

    def run(self):
        # results records
        agt_rewards = [[] for i in range(self.player_num)]
        agt_avg_rewards = [[] for i in range(self.player_num)]
        agt_bids = [[] for i in range(self.player_num)]
        estimate_time, revenue_list, avg_revenue_list = [], [], []
        saved_agt_strategies = []
        expl_lists = {key:[] for key in ['最小值', '最大值', '平均值']}
        # record per round
        last_bids = None
        for round in range(self.max_rounds+1):
            bid_values = np.zeros(self.player_num, dtype=int)
            for agt_idx, agt in enumerate(self.agt_list):
                # update with bidding of previous round
                if last_bids is not None:
                    last_reward = rewards[agt_idx]
                    if self.full_info:
                        # also send market price and market num
                        others_bids = last_bids.copy()
                        others_bids[agt_idx] = -1
                        market_price = max(others_bids)
                        market_num = sum(others_bids == market_price)
                    else:
                        # send reward only
                        market_price = market_num = None
                    agt.update_policy(last_reward, market_price, market_num)
                # generate action for current round
                action = agt.generate_action()
                bid_values[agt_idx] = action
                # record bid
                agt_bids[agt_idx].append(action)

            last_bids = bid_values.copy()

            # submit agents' bidding value to mechanism
            allocation, payment = self.auction_func(bid_values=bid_values)
            rewards = (self.agt_values - payment) * allocation
            for idx in range(self.player_num):
                agt_rewards[idx].append(rewards[idx])
            revenue_list.append(payment.sum())

            # logging
            length = len(agt_rewards[0])
            if round != 0 and round % self.log_freq == 0:
                avg_rewards = [sum(agt_rewards[i]) /
                               length for i in range(self.player_num)]
                avg_revenue = sum(revenue_list)/length
                print(f'Round {round}: rewards=',
                      avg_rewards, 'revenue=', avg_revenue,)
                plot_weights(self.agt_list, round, self.args)
                saved_agt_strategies.append({f'买家 {agt_id}': agt.weights.copy() for agt_id, agt in enumerate(self.agt_list)})

            # estimation for evolution
            if round != 0 and round % self.estimate_freq == 0:
                # recent avg revenue
                avg_revenue_list.append(sum(revenue_list[-self.estimate_freq:]) / self.estimate_freq)
                # latest max expl
                for key, value in calculate_expl(self.agt_values, [agt.weights for agt in self.agt_list], self.mechanism).items():
                    expl_lists[key].append(value)
                for agt_idx in range(self.player_num):
                    # avg rewards in recent K rounds
                    agt_avg_rewards[agt_idx].append(sum(agt_rewards[agt_idx][-self.estimate_freq:]) / self.estimate_freq)
                estimate_time.append(round)

        # plot results
        # plot_bids_evolve(plt_bids, estimate_time, self.args)
        plot_expl(expl_lists, estimate_time, self.args)

        # last_utilitys = [agt_avg_rewards[i][-1] for i in range(self.player_num)]
        # last_revenue = avg_revenue_list[-1]
        # plot_all_utility(last_utilitys.tolist(), last_revenue, self.args, 
        #                  self.theortc_utilitys.tolist(), self.theortc_revenue)
        plot_agts_utility(agt_avg_rewards, estimate_time, self.args, self.theortc_utilitys)
        plot_revenue(avg_revenue_list, estimate_time, self.args, self.theortc_revenue)
        
        # return plot info
        evolve_info = {
            '出价策略演化': saved_agt_strategies,
            '策略可利用度演化': expl_lists,
            '买家效益演化': agt_avg_rewards,
            '卖家收入演化': avg_revenue_list,
            # '演化效益 & 收入': {'卖家': last_revenue, '买家': last_utilitys},
        }
        return evolve_info

    def sample_result(self):
        bid_profile = {}
        for agt_idx, agt in enumerate(self.agt_list):
            bid_profile[f'买家 {agt_idx}'] = agt.generate_action(test=True)
        allocation, payment = self.auction_func(np.array(list(bid_profile.values())))
        utility = (self.agt_values - payment) * allocation
        winner = allocation.argmax()
        price = payment[winner]
        result = {
            '买家价值': {f'买家 {idx}': v for idx,v in enumerate(self.agt_values)},
            '出价结果': bid_profile,
            '赢家': f'买家 {winner}',
            '商品价格': price,
            '各买家效益': {f'买家 {idx}': u for idx,u in enumerate(utility)},
            '理论效益': {'卖家': self.theortc_revenue, '买家': self.theortc_utilitys},
        }
        plot_seller_buyer_pie(utility.tolist(), price, self.args, self.theortc_utilitys.tolist(), self.theortc_revenue)
        
        return result
