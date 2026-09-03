from abc import ABC, abstractmethod

import numpy as np


class InferenceEngine(ABC):

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        ...