# model_track/stats/correlation/base.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class CorrelationStrategy(ABC):

    @abstractmethod
    def supports(self, x: pd.Series, y: pd.Series) -> bool:
        pass

    @abstractmethod
    def compute(self, x: pd.Series, y: pd.Series) -> Dict:
        pass