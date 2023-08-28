# Hyperparameters for active learning
QUERY_BATCH_SIZE = 30
MAX_NUM_QUERIES = 20

# Hyperparameters for torch and keras NN
# Set EARLY_STOPPING_UPDATES_STUCK = None to disable early stopping;
# Otherwise we use a part of the initial training set for validation.
# This will effectively result in a smaller training set but can help against overfitting.
# Training is stopped early if the last improvement on the validation set was more than EARLY_STOPPING_UPDATES_STUCK parameter updates ago.
EARLY_STOPPING_UPDATES_STUCK = None
TRAINING_EPOCH_MAX = 50
TRAINING_BATCHSIZE = 64

# Hyperparameters for estimation of the uncertainty of dropout networks
NUM_SAMPLES = 50

# Hyperparameters for Monte Carlo Network
DROPOUT_RATE = 0.045

# Seed to use for all random number generators
SEED = 42

# Whether model parameters are saved at each iteration
SAVE_MODELS = False
