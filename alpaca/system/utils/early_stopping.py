from copy import deepcopy

class EarlyStoppingStrategy:
    """
    Represents an early stopping strategy for model training.
    
    The strategy automatically stores and updates the model with the best validation loss.
    At every epoch one has to call the update function.

    The strategy also keeps track of the following stopping criterion(s):
        1. If the maximum number of epochs is reached
        2. If the last improvement of the validation loss was more than a given number of parameter updates / minibatches ago.

    We choose parameter updates instead of epochs here, as the training time and the learning progress per epoch strongly depends
    on the size of the dataset. The number of updates is however invariant to the size of the dataset and so it is a bit more reliable
    for judging whether a model has converged and has started overfitting.

    """
    def __init__(self, max_epoch: int, max_updates_stuck: int):
        """
        @param max_epoch : Model training is stopped if this number of epochs has been reached.
        @param max_updates_stuck : Model training is stopped if the last improvement of the validation
            loss was more than this many parameter updates / minibatches ago
        
        """
        self.max_epoch: int = max_epoch
        self.max_updates_stuck: int = max_updates_stuck
        self.best_loss: float = None
        self.best_model_weights: object = None
        self.current_epoch: int = 0
        self.current_update: int = 0
        self.best_loss_update: int = 0

    def update(self, model_weights: object, validation_loss: float, num_updates: int) -> None:
        """
        Updates the model parameters and stopping criteria. Call this after each epoch.

        @param model_weights : The current model parameter weights. This can be a temporary view of
            your current model parameters and you do not need to copy them.
        @param validation_loss : Validation loss of the model.
        @param num_updates : Number of parameter updates since last epoch.
        """
        self.current_epoch += 1
        self.current_update += num_updates
        if self.best_loss is None or validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_loss_update = self.current_update 
            self.best_model_weights = deepcopy(model_weights)
    
    @property
    def stopping_criterion_met(self) -> bool:
        """
        Whether the stopping criterion has been met.
        """
        if self.current_epoch >= self.max_epoch:
            return True
        if self.current_update - self.best_loss_update > self.max_updates_stuck:
            return True
        return False

    def reset(self):
        """
        Resets the strategy to its initial state. This also removes the currently stored model weights.
        """
        self.best_loss: float = None
        self.best_model_weights: object = None
        self.current_epoch: int = 0
        self.best_loss_update: int = 0
