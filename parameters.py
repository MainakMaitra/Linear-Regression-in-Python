class Parameters:

    def __init__(self,
                 batch_size=0,
                 n_epochs=1000,
                 shuffle=False,
                 holdout_size=0.2,
                 l2=0,
                 learning_rate=0.01,
                 decay=1,
                 standardize=False,
                 adagrad=False,
                 rmsprop=False,
                 adam=False):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.holdout_size = holdout_size
        self.l2 = l2
        self.learning_rate = learning_rate
        self.decay = decay
        self.standardize = standardize
        self.adagrad = adagrad
        self.rmsprop = rmsprop
        self.adam = adam

    def get_params(self):
        return {
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "shuffle": self.shuffle,
            "holdout_size": self.holdout_size,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "standardize": self.standardize,
            "adagrad": self.adagrad,
            "rmsprop": self.rmsprop,
            "adam": self.adam
        }

    def set_params(self, **params):
        for parameter, value in params:
            self.__setattr__(parameter, value)
        return self
