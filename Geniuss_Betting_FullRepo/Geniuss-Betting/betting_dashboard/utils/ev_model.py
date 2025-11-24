import math

def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american is None:
        return None
    american = float(american)
    if american > 0:
        return 1.0 + (american / 100.0)
    else:
        return 1.0 + (100.0 / abs(american))

def implied_prob_from_american(american: float) -> float:
    dec = american_to_decimal(american)
    if not dec or dec <= 1:
        return None
    return 1.0 / dec

def expected_value(prob: float, american: float, stake: float = 1.0) -> float:
    """Return EV in units of stake for a given win probability and American odds."""
    if prob is None or american is None:
        return None
    dec = american_to_decimal(american)
    win_return = (dec - 1.0) * stake
    lose_amount = stake
    return prob * win_return - (1 - prob) * lose_amount

def kelly_fraction(prob: float, american: float) -> float:
    """Kelly stake fraction given edge and odds."""
    if prob is None or american is None:
        return 0.0
    dec = american_to_decimal(american)
    b = dec - 1.0
    q = 1 - prob
    edge = (b * prob - q)
    if b <= 0 or edge <= 0:
        return 0.0
    return edge / b
