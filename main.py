from keras.datasets import mnist
import numpy as np
import time
import math

LEARN_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 5

class Act:
    SIG  = "sigmoid"
    LIN  = "linear"
    TANH = "tanh"
    
    def __init__(self, acts):
        self.acts = acts
        
    def sig(n):
        return 1 / (1 + math.e**-n)
    
    def value(self, n):
        match self.acts:
            case self.TANH: return np.tanh(n)
            case self.SIG: return Act.sig(n)
            case self.LIN: return n

    def deriv(self, n):
        match self.acts:
            case self.TANH: return 1 - np.tanh(n)**2
            case self.SIG: return Act.sig(n) * (1 - Act.sig(n))
            case self.LIN: return n

ACTS = Act(Act.TANH)

class Cost:
    QUAD = "quad"
    
    def __init__(self, cost):
        self.cost = cost

    def value(self, err):
        match self.cost:
            case self.QUAD: return err**2

    def deriv(self, err):
        match self.cost:
            case self.QUAD: return 2 * err

COST = Cost(Cost.QUAD)

class Data:
    def __init__(
        self, 
        learn_rate=LEARN_RATE, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        act=ACTS, 
        cost=COST, 
        verbose=True
    ):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.acts = act
        self.cost = cost
        self.verbose = verbose

class Network:
    def __init__(self, form, data=Data()):
        self.form = form
        self.params = data
        self.LAYERS = len(form)
        
        self.weights = [np.random.rand(l2, l1) for (l1, l2) in zip(form[:-1], form[1:])]
        self.w_errs  = [np.zeros((l2, l1)) for (l1, l2) in zip(form[:-1], form[1:])]
        self.w_acc   = [np.zeros((l2, l1)) for (l1, l2) in zip(form[:-1], form[1:])]
        self.acts    = [np.zeros(l) for l in form]
        self.biases  = [np.zeros(l) for l in form[1:]]
        self.b_errs  = [np.zeros(l) for l in form[1:]]
        self.b_acc   = [np.zeros(l) for l in form[1:]]
        self.sums    = [np.zeros(l) for l in form[1:]]
        self.errs    = [np.zeros(l) for l in form[1:]]
        
    def __str__(self):
        w_shape = [w.shape for w in self.weights]
        b_shape = [b.shape for b in self.biases]
        a_shape = [a.shape for a in self.acts]
        s_shape = [s.shape for s in self.sums]

        return (f'Network{self.form} {{ \t\nweights: {w_shape}, \t\nbiases: {b_shape}, \t\nactivations: {a_shape}, \t\nsums: {s_shape} \n}}')

    def act(self, n):
        '''Applies non-linearity function to value `n`'''
        
        return self.params.acts.value(n)

    def d_act(self, n):
        '''Applies non-linearity derivative to value `n`'''
        
        return self.params.acts.deriv(n)

    def cost(self, delta):
        '''Applies cost function to `delta = Y - X`'''
        
        return self.params.cost.value(delta)

    def d_cost(self, delta):
        '''Applies cost derivative to `delta = Y - X`'''
        
        return self.params.cost.deriv(delta)

    def clear_prop(self):
        '''Resets propagation matrices to zero'''
        
        for i in range(self.LAYERS - 1):
            self.acts[i+1].fill(0)       
            self.sums[i].fill(0)
            self.errs[i].fill(0)
            self.w_errs[i].fill(0)
        
    def clear_accum(self):
        '''Resets accumulation matrices to zero'''
        
        for i in range(self.LAYERS - 1):
            self.b_acc[i].fill(0)
            self.w_acc[i].fill(0)

    def forward_prop(self, X):
        '''Forward passes an input, predicting an output'''
        
        self.acts[0] = X

        for l in range(self.LAYERS-1):
            np.matmul(
                self.weights[l], 
                self.acts[l], 
                out=self.sums[l])
            
            self.sums[l] += self.biases[l]

            self.acts[l+1] = self.act(self.sums[l]) 

        return self.acts[-1]

    def back_prop(self, X, Y):
        '''Computes the model's error given an 
        input and an expected output'''
        
        self.clear_prop()
        self.forward_prop(X)
        
        np.multiply(
            self.d_cost(Y - self.acts[-1]), 
            self.d_act(self.sums[-1]), 
            out=self.errs[-1])

        for l in range(1, self.LAYERS):
            np.matmul(
                self.errs[-l][:,None], 
                self.acts[-(l+1)][None,:], 
                out=self.w_errs[-l])

            if l == self.LAYERS - 1:
                break
            
            np.matmul(
                self.weights[-l].T, 
                self.errs[-l], 
                out=self.errs[-(l+1)])
            
            self.errs[-(l+1)] *= self.d_act(self.sums[-(l+1)])
    
    def accum_err(self):
        for i in range(self.LAYERS - 1):
            self.b_acc[i] += self.errs[i]
            self.w_acc[i] += self.w_errs[i]
            
    def apply_gradient(self, samples):
        lean_rate = self.params.learn_rate / samples
        
        for i in range(self.LAYERS - 1):
            self.weights[i] += self.w_acc[i] * lean_rate
            self.biases[i]  += self.b_acc[i] * lean_rate
      
    def train(self, Xs, Ys):
        '''Trains model on samples `Xs` with labels `Ys`'''
        
        n_samples = len(Xs)
        
        assert n_samples == len(Xs)
        
        for epoch in range(self.params.epochs):
            beg = time.time()
            
            self.clear_accum()
            accum_samples = 0
            
            for i in range(n_samples):
                self.back_prop(Xs[i], Ys[i])
                self.accum_err()
                
                accum_samples += 1  
                is_last = i + 1 == n_samples
                
                if accum_samples == self.params.batch_size or is_last:
                    self.apply_gradient(accum_samples)
                    self.clear_accum()              
                    accum_samples = 0
            
            if self.params.verbose:
                print(f'finished epoch {epoch} in {time.time() - beg}') 
                
    def predict(self, X):
        out = self.forward_prop(X)
        
        max = -math.inf
        idx = -1
        
        for i, n in enumerate(out):
            if n > max:
                max = n
                idx = i
                
        return idx, out
    
def one_hot(n):
    encoder = np.zeros(10)
    encoder[n] = 1
    return encoder

# nn = Network([3, 5, 3], Data(verbose=False, epochs=100, batch_size=1, learn_rate=.03, act=Act(Act.TANH)))
# X = np.array([-0.72, +0.99, -0.64])
# Y = np.array([+0.33, +0.03, +0.91])

# nn.train([X], [Y])
# p, out = nn.predict(X)

# print(out)
# print(f'error: {Y - out}')

network = Network([784, 300, 100, 10], Data(epochs=2, learn_rate=0.01, batch_size=16, act=Act(Act.SIG)))
                         
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = [np.concatenate(x) / 255 for x in train_x]
test_x  = [np.concatenate(x) / 255 for x in test_x] 
train_y = [one_hot(y) for y in train_y]
test_y  = [one_hot(y) for y in test_y]

network.train(train_x, train_y)

num, out = network.predict(train_x[0])

print(f'predicted: {num}!')
print(f'distribution: {out}')