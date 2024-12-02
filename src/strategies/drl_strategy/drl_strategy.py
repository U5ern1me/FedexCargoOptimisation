import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .drl_utils import DRL_Model
from strategies.strategy import Strategy

from utils.io import load_config

config = load_config(os.path.join(os.path.dirname(__file__), "drl.config"))


class DRLStrategy(Strategy):
    def solve(self):
        model = DRL_Model()
        model.train(self.ulds, self.packages, config)
