# built-in libraries
from dataclasses import dataclass

# internal classes
from MIW_Finance.src.derivative import Derivative


@dataclass
class Option(Derivative):

    strike_price: float        # strike price (dollars)
    time_to_expiry: float        # time to expiry (years)
    name: str

    @property
    def K(self):
        return self.strike_price
    @K.setter
    def K(self, val):
        self.strike_price = val

    @property
    def T(self):
        return self.time_to_expiry
    @T.setter
    def T(self, val):
        self.time_to_expiry = val

    def __copy__(self):
        new_option = Option(underlying=self.underlying,
                            strike_price=self.K,
                            time_to_expiry=self.T,
                            name=self.name)
        return new_option

