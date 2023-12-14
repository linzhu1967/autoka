class ValidationSaver:

    def __init__(self, use_loss, last_loss=10e9):
        """loss: boolean indicating if we monitor loss (True) or metric (False)"""
        self.use_loss = use_loss
        self.best = last_loss if use_loss else 0
        self.fn = lambda x, y: x < y if use_loss else x > y

    def __call__(self, val_perf, trainer, epoch):
        if self.fn(val_perf, self.best):
            # => improvement
            self.best = val_perf
            trainer.save_checkpoint(epoch=epoch, perf=val_perf, is_best=True)
