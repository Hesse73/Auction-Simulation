import os
import argparse
import random
import json
import numpy as np
from Env import PrivateAuction

parser = argparse.ArgumentParser()

# setting选取
parser.add_argument('-m', '--mechanism', type=str, default='second_price',
                    choices=['first_price', 'second_price', '1', '2'])
parser.add_argument('--algo', type=str, default='MWU', choices=['MWU'])
parser.add_argument('-v', '--agt_values', nargs='+',
                    type=int, default=[99, 49])
parser.add_argument('-i', '--full_info', type=int, default=1)

# 训练过程
parser.add_argument('--estimate_freq', type=int, default=50)
parser.add_argument('--log_freq', type=int, default=500)
parser.add_argument('-r', '--max_rounds', type=int, default=5000)
parser.add_argument('--folder_name', type=str, default='MWU_fixed')
parser.add_argument('--raw_plot', type=bool, default=True)

# 算法参数
parser.add_argument('--lr', type=float, default=0.01)


def prepare_args(args):
    # prepare args
    
    if args.mechanism == '2':
        args.mechanism = 'second_price'
    if args.mechanism == '1':
        args.mechanism = 'first_price'
        
    args.valuation_range = max(args.agt_values)
    args.player_num = len(args.agt_values)
    args.bidding_range = args.valuation_range
    
    # args.max_rounds = (int(args.player_num / 2) + 1) * 5000
    args.log_freq = int(args.max_rounds / 10)
    args.estimate_freq = int(args.max_rounds / 100)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    args = parser.parse_args()
    prepare_args(args)

    args.folder_name = f'{args.mechanism}_{args.agt_values}'

    # set seed
    random.seed(42)
    np.random.seed(42)

    env = PrivateAuction(args=args)
    evolution_info = env.run()
    
    sample_info = env.sample_result()
    print(sample_info)
    
    all_info = {'演化过程数据：': evolution_info,
                '最终出价 & 拍卖结果:': sample_info}
    path = os.path.join('./results', args.folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
        
    json.dump(all_info, open(os.path.join(path, 'all_info.json'), 'w', encoding='utf8'), cls=NpEncoder, ensure_ascii=False)
