import math


class CyclicAnnealing:
    def __init__(
            self,
            min_beta: float = 0.001,
            max_beta: float = 10.,
            cycle_length: int = 20,
            max_epoch: int = 0,
            min_epoch: int = 0,
    ):
        self.beta = min_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.cycle_length = cycle_length

    def step(self, current_epoch: float):
        progress_in_cycle = current_epoch % self.cycle_length / self.cycle_length
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_in_cycle))
        new_beta = self.min_beta + cosine_decay * (self.max_beta - self.min_beta)

        if current_epoch % self.cycle_length == 0:
            print(f"\nNew cycle started, beta weight updated: {self.beta:.4f} -> {new_beta:.4f}\n")

        self.beta = new_beta
