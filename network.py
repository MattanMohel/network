from hyperdata import Data
import numpy as np
import math
import time
import random

class Network:
    def __init__(self, form, data=Data()):
        self.form = form
        self.params = data
        self.LAYERS = len(form)
        
        self.weights = [self.init_weights(l2, l1) for (l1, l2) in zip(form[:-1], form[1:])]
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

        return \
            f'''Network{self.form} {{ 
                weights: {w_shape}, 
                biases: {b_shape}, 
                activations: {a_shape}, 
                sums: {s_shape}
            }}'''

    def init_weights(self, row, col):
        return self.params.weight_init.random_weights(row, col)
    
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

    def loss(self, Xs, Ys):
        '''Computes the loss of the model'''

        correct = 0
        for x, y in zip(Xs, Ys):
            n, _ = self.predict(x) 
            if n == np.argmax(y):
                correct += 1

        return correct / len(Xs)
      
    def train(self, Xs, Ys):
        '''Trains model on samples `Xs` with labels `Ys`'''
        
        n_samples = len(Xs)
        
        assert n_samples == len(Xs)
        
        for epoch in range(self.params.epochs):
            if self.params.shuffle:
                seed = time.time()
                random.Random(seed).shuffle(Xs)
                random.Random(seed).shuffle(Ys)
            
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
                loss = self.loss(Xs, Ys)
                print(f'epoch accuracy: {loss}')
                
                
    def predict(self, X):
        out = self.forward_prop(X)
        
        max = -math.inf
        idx = -1
        
        for i, n in enumerate(out):
            if n > max:
                max = n
                idx = i
                
        return idx, out
    