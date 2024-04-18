import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_style('darkgrid')
font_options = {
    'family' : 'serif', # 设置字体家族
    'serif' : 'simsun', # 设置字体
}
plt.rc('font',**font_options)

def plot_bids_evolve(plt_bids, estimate_time, args):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{args.player_num}_bidders_{args.mechanism}_bid_policy_evolution'
    for agt_id, bid_infos in enumerate(plt_bids):
        y = np.array(bid_infos['mean'])
        y_min = y - np.array(bid_infos['min'])
        y_max = np.array(bid_infos['max']) - y
        plt.errorbar(estimate_time, y, yerr=[
                     y_min, y_max], label=f'bidder_{agt_id}')
    plt.xlabel('round')
    plt.ylabel('bid')
    plt.legend()
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()


def plot_all_utility(utilitys:list, revenue:float, args, theortc_utilitys:list, theortc_revenue:float):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = 'seller_buyer_bar'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    bidder_num = len(utilitys)
    # def show_pie(axis, utilitys:list, revenue:float):
    #     labels = ['seller'] + [f'bidder {i}' for i in range(bidder_num)]
    #     sizes = [revenue] + utilitys
    #     explode = [0.1] + [0] * bidder_num
    #     axis.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    # show_pie(ax1, utilitys=last_utilitys, revenue=last_revenue)
    # show_pie(ax2, theortc_utilitys, theortc_revenue)
    def show_bar(axis, utilitys:list, revenue:float):
        labels = ['卖家'] + [f'买家 {i}' for i in range(bidder_num)]
        value = [revenue] + utilitys
        colors = [plt.cm.tab10.colors[i] for i in range(bidder_num + 1)]
        axis.set_ylim(0, sum(value)+10)
        # axis.bar(labels, value)
        axis.bar(labels, value, label=labels, color=colors, alpha=0.8)
        axis.legend()
    show_bar(ax1, utilitys, revenue)
    show_bar(ax2, theortc_utilitys, theortc_revenue)
    ax1.set_title('模拟收益结果')
    ax2.set_title('计算收益结果')
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()


def plot_seller_buyer_pie(utilitys:list, revenue:float, args, theortc_utilitys:list, theortc_revenue:float):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = 'seller_buyer_pie'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    def show_pie(axis, utilitys:list, revenue:float):
        labels = ['卖家', '买家']
        sizes = [revenue, sum(utilitys)]
        explode = [0.1, 0]
        axis.pie(sizes, explode=explode, labels=labels, autopct=lambda x:'{:.1f}'.format(x), shadow=True, startangle=90)
    show_pie(ax1, utilitys, revenue)
    show_pie(ax2, theortc_utilitys, theortc_revenue)
    ax1.set_title('模拟收益结果')
    ax2.set_title('理论收益结果')
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()

def plot_weights(agt_list, round, args):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{args.player_num}_bidders_round_{round}_weights'
    for agt_idx, agt in enumerate(agt_list):
        plt.bar([x + 0.01 * args.valuation_range * agt_idx for x in range(agt.range)],
                agt.weights, label=f"买家 {agt_idx}")
    plt.xlabel('出价值')
    plt.ylabel('概率')
    plt.title('各买家 出价-概率图')
    plt.legend()
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()


def plot_agts_utility(agt_avg_rewards, estimate_time, args, theortc_utilitys):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'bidders_rewards_plot'
    for agt_idx, rewards in enumerate(agt_avg_rewards):
        plt.plot(estimate_time, rewards,
                 label=f'买家 {agt_idx}')
    plt.xlabel('轮数')
    plt.ylabel('效益值')
    plt.title('各买家效益演化过程')
    plt.legend()
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()


def plot_revenue(avg_revenue_list, estimate_time, args, theortc_revenue):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'seller_revenue_plot'
    plt.plot(estimate_time, avg_revenue_list)
    # plt.axhline(y=theortc_revenue, linestyle='dotted',
    #             label='theoretical revenue')
    plt.xlabel('轮数')
    plt.ylabel('收入额')
    plt.title('卖家收入演化过程')
    # plt.legend()
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()


def plot_expl(expl_lists, estimate_time, args):
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'expl_plot'
    for key, expl_l in expl_lists.items():
        plt.plot(estimate_time, expl_l, label=f"可利用度:{key}")
    plt.xlabel('轮数')
    plt.ylabel('可利用度')
    plt.title('各买家策略可利用度演化')
    # plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(path, filename))
    #plt.show()
    plt.close()
