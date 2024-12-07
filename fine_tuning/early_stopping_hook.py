from mmcv.runner import Hook

class EarlyStoppingHook(Hook):
    def __init__(self, patience=5, metric='bbox_mAP', interval=1):
        self.patience = patience
        self.metric = metric
        self.interval = interval
        self.best_score = -float('inf')
        self.num_bad_epochs = 0

    def after_val_epoch(self, runner):
        # Get the current score from validation metrics
        score = runner.log_buffer.output.get(self.metric, None)
        if score is None:
            return

        # Check if the score has improved
        if score > self.best_score:
            self.best_score = score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Stop training if patience is exceeded
        if self.num_bad_epochs >= self.patience:
            runner.should_stop = True