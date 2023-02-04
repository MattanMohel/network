import math
import numpy as np

def SIGMOID(n):
    return 1 / (1 + math.e**-n)

class Act:
    SIG  = 0
    LIN  = 1
    TANH = 2
    
    def __init__(self, acts):
        self.acts = acts
            
    def value(self, n):
        '''Applies non-linearity function to `n`'''
        
        match self.acts:
            case self.TANH: return np.tanh(n)
            case self.SIG: return SIGMOID(n)
            case self.LIN: return n

    def deriv(self, n):
        '''Applies non-linearity derivative to `n`'''
        
        match self.acts:
            case self.TANH: return 1 - np.tanh(n)**2
            case self.SIG: return SIGMOID(n) * (1 - SIGMOID(n))
            case self.LIN: return n

class Cost:
    QUAD = 0
    
    def __init__(self, cost):
        self.cost = cost

    def value(self, err):
        '''Applies cost function to `err`'''
        
        match self.cost:
            case self.QUAD: return err**2

    def deriv(self, err):
        '''Applies cost derivative to `err`'''
        
        match self.cost:
            case self.QUAD: return 2 * err

class WeightInit:
    def __init__(self, min, max, norm):
        self.min = min
        self.max = max
        self.norm = norm

    def random_weights(self, row, col):
        weights = np.random.random_sample((row, col))
        return ((self.max - self.min) * weights + self.min) / self.norm

WEIGHT_INIT = WeightInit(-1, +1, 5)
SIG  = Act(Act.SIG)
LIN  = Act(Act.LIN)
TANH = Act(Act.TANH)
QUAD = Cost(Cost.QUAD)
LEARN_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 5

class Data:
    def __init__(
        self, 
        learn_rate=LEARN_RATE, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        act=TANH, 
        cost=QUAD, 
        weight_init=WEIGHT_INIT,
        shuffle=True,
        verbose=True
    ):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.acts = act
        self.cost = cost
        self.weight_init = weight_init
        self.shuffle = shuffle
        self.verbose = verbose
