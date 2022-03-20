
# built-in libraries
from dataclasses import dataclass


@dataclass
class Market:

    risk_free_interest_rate: float
    name: str

    @property
    def r(self):
        return self.risk_free_interest_rate
    @r.setter
    def r(self, val):
        self.risk_free_interest_rate = val
