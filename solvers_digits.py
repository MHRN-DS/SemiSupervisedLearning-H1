import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
#from heap import Heap, max_priority_generator


class SemiSupervisedSolver:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled = None, **kwargs):
        """Base class for semi-supervised learning solvers using similarity-based approach.

        This class implements the base functionality for solving semi-supervised learning problems
        using similarity matrices between labeled and unlabeled data points. The objective is to
        propagate labels from labeled data points to unlabeled ones based on their similarities.

        Parameters
        ----------
        X_labeled : numpy.ndarray, shape (n_labeled, n_features)
            Feature matrix of labeled data points
        y_labeled : numpy.ndarray, shape (n_labeled,)
            Labels for the labeled data points
        X_unlabeled : numpy.ndarray, shape (n_unlabeled, n_features)
            Feature matrix of unlabeled data points
        y_unlabeled : numpy.ndarray, shape (n_unlabeled,), optional
            Initial guess for unlabeled data points. If None, initialized as random
        Notes
        -----
        The similarity matrices are computed using RBF kernel with gamma=1.0
        """

        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled


        self.l = y_labeled.shape[0]
        self.u = X_unlabeled.shape[0]

        self.W_lu = rbf_kernel(X_labeled, X_unlabeled, gamma=1.0)
        self.W_uu = rbf_kernel(X_unlabeled, X_unlabeled, gamma=1.0)

        if y_unlabeled is not None:
            self.y_unlabeled = y_unlabeled
        else:
            self.y_unlabeled = np.random.rand(X_unlabeled.shape[0])

    def step(self):
        """Single update step. To be implemented by subclasses."""
        raise NotImplementedError

    def full_grad(self): # ATTENTION FOR FACTOR
        return 2 * (self.W_lu.sum(axis=0) * self.y_unlabeled + self.W_uu.sum(axis=0) * self.y_unlabeled - (self.W_lu.T @ self.y_labeled + self.W_uu.T @ self.y_unlabeled))
        
    def compute_loss(self, normalized=False):
        """
        Compute the graph-based Laplacian loss with optional normalization.
        
        Parameters:
        -----------
        normalized : bool, optional (default=True)
            If True, normalizes loss by number of terms to make it scale-invariant
            
        Returns:
        --------
        float
            The computed loss value
        """
        # Term 1: Labeled-unlabeled consistency (vectorized)
        term1 = np.mean(self.W_lu * (self.y_unlabeled[None,:] - self.y_labeled[:,None])**2)
        
        # Term 2: Unlabeled-unlabeled smoothness (vectorized)
        diff = self.y_unlabeled[:,None] - self.y_unlabeled[None,:]
        term2 = 0.5 * np.mean(self.W_uu * (diff**2))
        
        if normalized:
            # Normalize by number of terms using /= operator
            term1 /= (self.l * self.u)  # Normalize by (labeled points × unlabeled points)
            term2 /= (self.u ** 2)      # Normalize by (unlabeled points × unlabeled points)
            
        return term1 + term2

class GradientDescentSolver(SemiSupervisedSolver):
    def __init__(self, *args, lr=0.01, momentum = 0.0, nesterov = False, **kwargs):
        """Gradient Descent solver for semi-supervised learning.

        Implements standard gradient descent optimization for label propagation.

        Parameters
        ----------
        *args
            Arguments to pass to parent SemiSupervisedSolver
        lr : float, default=0.01
            Learning rate for gradient descent updates

        momentum : float, default=0.0
            Momentum coefficient (β). Set >0 for Heavy Ball GD

        nesterov : bool, default=False
            If True, uses Nesterov Accelerated Gradient (NAG) update.

        Notes
        -----
        Updates all unlabeled points simultaneously using full gradient computation
        """
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = 0


    def step(self):
        grad = self.full_grad()

        if isinstance(self.velocity, int):
            self.velocity = 0 * grad


        if self.nesterov and self.momentum > 0: # if False we compute Heavy Ball or Regular Gradient Descent with/without accelerated step size
            # Compute look-ahead position, since we check beforehand computing the gradient 
            lookahead = self.y_unlabeled + self.momentum * self.velocity 
            
            # Compute gradient at look-ahead point
            grad = 2 * (self.W_lu.sum(axis=0) * lookahead + self.W_uu.sum(axis=0) * lookahead - (self.W_lu.T @ self.y_labeled + self.W_uu.T @ lookahead))

            # Update velocity
            self.velocity = self.momentum * self.velocity - self.lr * grad

            # move parameters
            self.y_unlabeled += self.velocity

        else:
            self.velocity = self.momentum * self.velocity - self.lr *grad

            self.y_unlabeled += self.velocity


class BCGDSolver(SemiSupervisedSolver):
    def __init__(self, *args, lr=0.01, cache=True, refresh_rate=50, **kwargs):
        """"
        Block Coordinate Gradient Descent solver for semi-supervised learning.
    
        Implements BCGD optimization with optional gradient caching for efficiency.

        Parameters
        ----------
        *args
            Arguments to pass to parent SemiSupervisedSolver
        lr : float, default=0.01
            Learning rate for gradient updates
        cache : bool, default=True
            Whether to cache and update gradients (True) or recompute them with full gradient (False)
        """
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.cache = cache
        self.grad = self.full_grad()

        self.W_lu_colsum = self.W_lu.sum(axis=0)
        self.W_uu_colsum = self.W_uu.sum(axis=0)   

        self.step_counter = 0   
        self.refresh_rate = refresh_rate

    def step(self):
        if not self.cache:
            grad = self.full_grad()
            idx = np.argmax(np.abs(grad))
            self.y_unlabeled[idx] -= self.lr * grad[idx]
            self.y_unlabeled[idx] = np.clip(self.y_unlabeled[idx], 0, 1)
        else:
            # this else block is for BCGD with cache
            
            idx = np.argmax(np.abs(self.grad))  # Use informative index
            delta = -self.lr * self.grad[idx]
            self.y_unlabeled[idx] += delta
            self.y_unlabeled[idx] = np.clip(self.y_unlabeled[idx], 0, 1)

            #self.grad = grad

            self.grad -= delta * self.W_uu[:, idx]
            self.grad[idx] += delta * (self.W_lu_colsum[idx] + self.W_uu_colsum[idx])

            # Periodic full refresh of cache
            self.step_counter += 1
            if self.step_counter % 50 == 0:
                self.grad = self.full_grad()


class CoordinateMinimizationSolver(SemiSupervisedSolver):
    def __init__(self, *args, **kwargs):
        """
        Coordinate Minimization solver for semi-supervised learning.

        Implements cyclic coordinate minimization by optimizing one coordinate
        at a time in sequence.

        Parameters
        ----------
        *args
            Arguments to pass to parent SemiSupervisedSolver

        Attributes
        ----------
        k : int
            Counter for tracking current coordinate being updated

        Notes
        -----
        Updates coordinates cyclically by computing the optimal value for each
        coordinate while keeping others fixed.
        """
        super().__init__(*args, **kwargs)
        self.k = 0

    def step(self):
        j = self.k % self.u
        self.y_unlabeled[j] = (self.W_lu[:, j].T @ self.y_labeled + self.W_uu[:, j].T @ self.y_unlabeled - self.W_uu[j, j] * self.y_unlabeled[j]) / \
                              ((self.W_lu[:, j].sum() + self.W_uu[:, j].sum() - self.W_uu[j, j]))
        self.k += 1
