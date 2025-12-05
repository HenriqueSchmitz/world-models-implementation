from abc import ABC, abstractmethod

class EarlyStopper(ABC):
    @abstractmethod
    def __call__(self, validation_loss: float) -> bool:
        """
        Returns True if the training should be stopped.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
class EarlyStopping(EarlyStopper):
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, validation_loss: float):
        padded_validation_loss = validation_loss * (1 - self.min_delta)
        if padded_validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.tolerance:
            return True
        return False