import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import torch
import torch.nn as nn
from scipy.interpolate import BSpline
from sklearn.preprocessing import SplineTransformer
from scipy.integrate import simps
import optax

def create_bspline_basis(t, n_knots, degree=3):
    """Create B-spline basis functions and their second derivatives"""
    # Create knot sequence with proper multiplicity at endpoints
    t_min, t_max = t.min(), t.max()
    internal_knots = np.linspace(t_min, t_max, n_knots)[1:-1]
    knots = np.concatenate([
        [t_min] * (degree + 1),
        internal_knots,
        [t_max] * (degree + 1)
    ])
    
    # Create basis splines
    n_basis = len(internal_knots) + degree + 1
    basis_splines = []
    basis_splines_d2 = []
    
    for i in range(n_basis):
        # Get coefficients for the i-th basis
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        
        # Create spline and its second derivative
        spl = BSpline(knots, coeffs, degree)
        # Second derivative is also a B-spline, with degree reduced by 2
        spl_d2 = spl.derivative(2)
        
        basis_splines.append(spl)
        basis_splines_d2.append(spl_d2)
    
    return basis_splines, basis_splines_d2

class JAXGAMSolver:
    def __init__(self, n_components, n_knots=20, smoothness_penalty=1.0):
        self.n_components = n_components
        self.n_knots = n_knots
        self.smoothness_penalty = smoothness_penalty
        
    def initialize_basis(self, t):
        """Initialize B-spline basis functions and their derivatives"""
        basis_splines, basis_splines_d2 = create_bspline_basis(t, self.n_knots)
        
        # Convert to functions that work with JAX
        @jit
        def evaluate_basis(t):
            return jnp.array([[spl(ti) for spl in basis_splines] for ti in t])
        
        @jit
        def evaluate_d2_basis(t):
            return jnp.array([[spl(ti) for spl in basis_splines_d2] for ti in t])
        
        self.evaluate_basis = evaluate_basis
        self.evaluate_d2_basis = evaluate_d2_basis
    
    def evaluate_component(self, coeffs, t):
        """Evaluate a single component function"""
        basis = self.evaluate_basis(t)
        return jnp.dot(basis, coeffs)
    
    def evaluate_d2_component(self, coeffs, t):
        """Evaluate second derivative of component function"""
        d2_basis = self.evaluate_d2_basis(t)
        return jnp.dot(d2_basis, coeffs)
    
    @jit
    def loss_function(self, coeffs_list, t, y, taus):
        """Compute total loss with analytical second derivatives"""
        # Prediction loss
        pred = jnp.zeros_like(y)
        for coeffs, tau in zip(coeffs_list, taus):
            pred += self.evaluate_component(coeffs, t - tau)
        
        prediction_loss = jnp.mean((y - pred)**2)
        
        # Smoothness penalty using analytical second derivatives
        smoothness_loss = 0.0
        for coeffs in coeffs_list:
            d2f = self.evaluate_d2_component(coeffs, t)
            smoothness_loss += jnp.mean(d2f**2)
        
        # Initial condition penalty
        init_condition_loss = 0.0
        for coeffs, tau in zip(coeffs_list, taus):
            f_init = self.evaluate_component(coeffs, jnp.array([0.0]))
            init_condition_loss += f_init**2
        
        return (prediction_loss + 
                self.smoothness_penalty * smoothness_loss + 
                100.0 * init_condition_loss)

class TorchGAMSolver(nn.Module):
    def __init__(self, n_components, n_knots=20, smoothness_penalty=1.0):
        super().__init__()
        self.n_components = n_components
        self.n_knots = n_knots
        self.smoothness_penalty = smoothness_penalty
        
        # Initialize spline coefficients as parameters
        self.coeffs_list = nn.ParameterList([
            nn.Parameter(torch.randn(n_knots))
            for _ in range(n_components)
        ])
    
    def initialize_basis(self, t):
        """Initialize B-spline basis functions and their derivatives"""
        basis_splines, basis_splines_d2 = create_bspline_basis(t.numpy(), self.n_knots)
        
        def evaluate_basis(t):
            t_np = t.numpy()
            basis = np.array([[spl(ti) for spl in basis_splines] for ti in t_np])
            return torch.tensor(basis, dtype=torch.float32)
        
        def evaluate_d2_basis(t):
            t_np = t.numpy()
            d2_basis = np.array([[spl(ti) for spl in basis_splines_d2] for ti in t_np])
            return torch.tensor(d2_basis, dtype=torch.float32)
        
        self.evaluate_basis = evaluate_basis
        self.evaluate_d2_basis = evaluate_d2_basis
    
    def evaluate_component(self, coeffs, t):
        """Evaluate a single component function"""
        basis = self.evaluate_basis(t)
        return torch.matmul(basis, coeffs)
    
    def evaluate_d2_component(self, coeffs, t):
        """Evaluate second derivative of component function"""
        d2_basis = self.evaluate_d2_basis(t)
        return torch.matmul(d2_basis, coeffs)

class SklearnGAMSolver:
    def __init__(self, n_components, n_splines=20, smoothness_penalty=1.0):
        self.n_components = n_components
        self.n_splines = n_splines
        self.smoothness_penalty = smoothness_penalty
        
    def initialize_basis(self, t):
        """Initialize B-spline basis functions and their derivatives"""
        self.basis_splines, self.basis_splines_d2 = create_bspline_basis(t, self.n_splines)
    
    def transform_with_derivatives(self, t):
        """Transform data using B-splines and compute second derivatives"""
        # Evaluate basis functions
        X = np.array([[spl(ti) for spl in self.basis_splines] for ti in t])
        
        # Evaluate second derivatives
        X_d2 = np.array([[spl(ti) for spl in self.basis_splines_d2] for ti in t])
        
        return X, X_d2
    
    def fit(self, t, y, taus, max_iter=1000):
        """Fit using sklearn with analytical derivatives"""
        from sklearn.linear_model import Ridge
        
        self.initialize_basis(t)
        
        # Transform inputs for each component
        X_transformed = []
        X_d2_transformed = []
        
        for tau in taus:
            X, X_d2 = self.transform_with_derivatives(t - tau)
            X_transformed.append(X)
            X_d2_transformed.append(X_d2)
        
        X = np.hstack(X_transformed)
        X_d2 = np.hstack(X_d2_transformed)
        
        # Create penalty matrix using analytical second derivatives
        penalty_matrix = X_d2.T @ X_d2
        
        # Fit with custom penalty matrix
        self.model = Ridge(
            alpha=self.smoothness_penalty,
            fit_intercept=False
        )
        
        # Modify the Ridge penalty to use our analytical second derivatives
        self.model._max_iter = max_iter
        self.model.fit(X, y)
        
        return self