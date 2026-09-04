from abc import ABC, abstractmethod

import numpy as np


class InferenceEngine(ABC):

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        ...