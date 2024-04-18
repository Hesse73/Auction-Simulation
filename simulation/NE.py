import numpy as np


def calculate_NE(agt_values, mechanism='second_price'):
    bidder_num = len(agt_values)
    agt_values = np.array(agt_values)
    max_value = max(agt_values)
    max_num = sum(agt_values == max_value)
    sorted_values = sorted(agt_values)

    if mechanism == 'second_price':
        # assume truthful bidding
        if max_num >= 2:
            utilitys = np.zeros_like(agt_values)
            revenue = max_value
        else:
            second_value = sorted_values[-2]
            utilitys = np.where(agt_values == max_value,
                                max_value - second_value, 0)
            revenue = second_value
    else:
        if max_num >= 3:
            utilitys = np.where(agt_values == max_value, 1/max_num, 0)
            revenue = max_value - 1
        elif max_num == 2:
            third_value = sorted_values[-3]
            if bidder_num == 2 or max_value > third_value + 1:
                utilitys = np.where(agt_values == max_value, 1, 0)
                revenue = max_value - 2
            else:
                utilitys = np.where(agt_values == max_value, 0.5, 0)
                revenue = max_value - 1
        else:
            second_value = sorted_values[-2]
            utilitys = np.where(agt_values == max_value,
                                max_value - second_value - 1, 0)
            revenue = second_value + 1

    return utilitys, revenue


def calculate_expl(agt_values, strategies, mechanism='second_price'):
    agt_values = np.array(agt_values)
    max_range = max(agt_values) + 1
    bidder_num = len(agt_values)
    # padding
    strategies = np.array([np.pad(x, (0, max_range - len(x)),
                          'constant', constant_values=(0, 0)) for x in strategies])

    ## 1. calculate market price
    # cumulative distribution
    cumulative_bids = np.cumsum(strategies, axis=-1)  # N*V
    # others' cumulative bid
    others_cumbids = np.repeat(np.expand_dims(
        cumulative_bids, axis=0), bidder_num, axis=0)  # N*N*V
    self_mask = np.arange(bidder_num) == np.arange(
        bidder_num).reshape(-1, 1)  # N*N
    self_mask = np.expand_dims(
        self_mask, axis=-1).repeat(max_range, axis=-1)  # N*N*V
    others_cumbids[self_mask] = 1
    market_cdf = np.prod(others_cumbids, axis=1)  # N*V
    # market price distribution
    tmp = np.zeros_like(market_cdf)
    tmp[:, 1:] = market_cdf[:, :-1]
    market_pdf = market_cdf - tmp  # N*V
    # reset negative to 0 (might occur due to precision error)
    market_pdf[market_pdf < 0] = 0

    ## 2. calcualte utility
    value = agt_values.reshape(
        [-1, 1, 1]).repeat(max_range, axis=1).repeat(max_range, axis=2)
    market = np.arange(max_range).reshape(
        [1, -1, 1]).repeat(bidder_num, axis=0).repeat(max_range, axis=2)
    bid = np.arange(max_range).reshape(
        [1, 1, -1]).repeat(bidder_num, axis=0).repeat(max_range, axis=1)

    # utility matrix: N*V*V
    if mechanism == 'second_price':
        utility_n_m_b = (value - market) * (bid > market)
    else:
        utility_n_m_b = (value - bid) * (bid > market)

    ## 3. exploitability
    # expectation over market price distribution
    utility_n_b = np.matmul(market_pdf.reshape(
        [bidder_num, 1, max_range]), utility_n_m_b).squeeze(1)  # N*1*V @ N*V*V -> N*1*V
    # expectation over bid distribution
    # N*V * N*V -- sum over last dim -> N
    utility_n = (utility_n_b * strategies).sum(axis=-1)

    best_utility = utility_n_b.max(axis=1)  # N
    expl = (best_utility - utility_n) / max(agt_values)
    measurement = {
        '最小值': expl.min(),
        '最大值': expl.max(),
        '平均值': expl.mean()
    }
    
    return measurement
