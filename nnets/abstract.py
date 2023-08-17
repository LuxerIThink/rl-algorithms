from abc import ABC, abstractmethod


class NNet(ABC):
    @abstractmethod
    def forward(self, x):
        pass
