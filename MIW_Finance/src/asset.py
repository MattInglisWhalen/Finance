
from dataclasses import dataclass


@dataclass
class Asset:

    name: str
    curr_price: float           # (dollars)
    curr_volatility: float      # (per root-annum)
    curr_drift_rate: float = 0  # geometric drift rate (per annum)

    @property
    def S0(self):
        return self.curr_price
    @S0.setter
    def S0(self, val):
        self.curr_price = val

    @property
    def mu(self):
        return self.curr_drift_rate
    @mu.setter
    def mu(self, val):
        self.curr_drift_rate = val

    @property
    def sigma(self):
        return self.curr_volatility
    @sigma.setter
    def sigma(self, val):
        self.curr_volatility = val
