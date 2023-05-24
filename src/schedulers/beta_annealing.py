class BetaAnnealing:
    def __init__(
            self,
            min_beta: float = 0.001,
            max_beta: float = 10.,
            start_epoch: int = 0,
            end_epoch: int = 10,
    ):
        self.beta = min_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def step(self, current_epoch: float):
        if current_epoch > self.end_epoch:
            self.beta = self.max_beta
        elif current_epoch >= self.start_epoch:
            old_beta = self.beta
            frac_epochs = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            new_beta = self.min_beta + frac_epochs * (self.max_beta - self.min_beta)

            print(f"\nBeta weight updated: {old_beta:.4f} -> {new_beta:.4f}\n")
            self.beta = new_beta
            if current_epoch == self.end_epoch:
                print(f"\nBeta maximum value reached: {self.beta}\n")
