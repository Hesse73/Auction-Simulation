import numpy as np


def first_price_auction(bid_values):
    """
    First price auction
    bid_values (N*bids) -> allocation & payment
    all stored as numpy array
    """
    win_bidder = np.random.choice(
        np.flatnonzero(bid_values == bid_values.max()))
    allocation = np.arange(len(bid_values)) == win_bidder

    payment = np.zeros(len(bid_values))
    payment[win_bidder] = bid_values.max()

    return allocation, payment


def second_price_auction(bid_values):
    """
    Second price auction
    bid_values (N*bids) -> allocation & payment
    all store as numpy array
    """
    win_bidder = np.random.choice(
        np.flatnonzero(bid_values == bid_values.max()))
    allocation = np.arange(len(bid_values)) == win_bidder

    payment = np.zeros(len(bid_values))
    payment[win_bidder] = np.partition(bid_values, -2)[-2]

    return allocation, payment
