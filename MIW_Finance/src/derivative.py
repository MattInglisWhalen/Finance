
# built-in libraries
from dataclasses import dataclass

# internal classes
from MIW_Finance.src.asset import Asset


@dataclass
class Derivative:

    underlying: Asset
    name: str
