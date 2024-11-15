import numpy as np
from typing import Dict, Optional, Tuple
from scipy import linalg

class SteadyStateCalculator:
    """Calculate steady-state probabilities for hypercube model."""
    
    def __init__(self, N: int, lambda_rate: float, mu_rate: float):
        """Initialize calculator with system parameters.
        
        Args:
            N (int): Number of units
            lambda_rate (float): Call arrival rate
            mu_rate (float): Service rate
        """
        self.N = N
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.rho = lambda_rate / (N * mu_rate)
        
    def solve_steady_state(self, A: np.ndarray, method: str = 'direct') -> np.ndarray:
        """Solve for steady-state probabilities using specified method.
        
        Args:
            A (numpy.ndarray): Transition rate matrix
            method (str): Solution method ('direct', 'iterative', or 'mm_n')
            
        Returns:
            numpy.ndarray: Steady-state probabilities
        """
        if method == 'direct':
            return self._solve_direct(A)
        elif method == 'iterative':
            return self._solve_iterative(A)
        elif method == 'mm_n':
            return self._solve_mm_n()
        else:
            raise ValueError(f"Unknown solution method: {method}")
    
    def _solve_direct(self, A: np.ndarray) -> np.ndarray:
        """Solve using direct method (matrix inversion).
        
        Args:
            A (numpy.ndarray): Transition rate matrix
            
        Returns:
            numpy.ndarray: Steady-state probabilities
        """
        num_states = A.shape[0]
        
        # Replace last row with normalization equation
        A_mod = A.copy()
        A_mod[-1] = 1.0
        
        # Create right-hand side
        b = np.zeros(num_states)
        b[-1] = 1.0
        
        # Solve system
        pi = linalg.solve(A_mod.T, b)
        
        return pi
    
    def _solve_iterative(self, A: np.ndarray, max_iter: int = 1000, 
                        tol: float = 1e-10) -> np.ndarray:
        """Solve using iterative method (Gauss-Seidel).
        
        Args:
            A (numpy.ndarray): Transition rate matrix
            max_iter (int): Maximum iterations
            tol (float): Convergence tolerance
            
        Returns:
            numpy.ndarray: Steady-state probabilities
        """
        num_states = A.shape[0]
        pi = np.ones(num_states) / num_states  # Initial guess
        
        for iteration in range(max_iter):
            pi_new = pi.copy()
            
            # Update each probability
            for j in range(num_states - 1):  # Skip last equation
                sum_term = 0
                for i in range(num_states):
                    if i != j:
                        sum_term += pi_new[i] * A[i,j]
                pi_new[j] = -sum_term / A[j,j]
            
            # Update last probability using normalization
            pi_new[-1] = 1.0 - np.sum(pi_new[:-1])
            
            # Check convergence
            if np.max(np.abs(pi_new - pi)) < tol:
                return pi_new
                
            pi = pi_new
            
        raise RuntimeError(f"Failed to converge after {max_iter} iterations")
    
    def _solve_mm_n(self) -> np.ndarray:
        """Solve using M/M/N queue formulas.
        
        Returns:
            numpy.ndarray: Steady-state probabilities
        """
        num_states = 2**self.N
        pi = np.zeros(num_states)
        
        # Compute p0
        sum_term = sum((self.N * self.rho)**n / np.math.factorial(n) 
                      for n in range(self.N))
        p0 = 1.0 / sum_term
        
        # Compute other probabilities
        for state in range(num_states):
            n = bin(state).count('1')  # Number of busy units
            if n <= self.N:
                pi[state] = p0 * (self.N * self.rho)**n / np.math.factorial(n)
                
        # Normalize
        pi = pi / np.sum(pi)
        
        return pi
    
    def compute_performance_measures(self, pi: np.ndarray) -> Dict[str, float]:
        """Compute steady-state performance measures.
        
        Args:
            pi (numpy.ndarray): Steady-state probabilities
            
        Returns:
            Dict[str, float]: Performance measures
        """
        measures = {}
        
        # Probability of all units busy
        all_busy_states = [i for i in range(len(pi)) 
                          if bin(i).count('1') == self.N]
        measures['p_all_busy'] = sum(pi[i] for i in all_busy_states)
        
        # Average number of busy units
        measures['avg_busy_units'] = sum(
            bin(i).count('1') * p for i, p in enumerate(pi)
        )
        
        # System utilization
        measures['utilization'] = measures['avg_busy_units'] / self.N
        
        # Probability of immediate service
        measures['p_immediate'] = 1 - measures['p_all_busy']
        
        return measures
    
    def verify_solution(self, A: np.ndarray, pi: np.ndarray, tol: float = 1e-8) -> bool:
        """Verify that solution satisfies balance equations.
        
        Args:
            A (numpy.ndarray): Transition rate matrix
            pi (numpy.ndarray): Steady-state probabilities
            tol (float): Verification tolerance
            
        Returns:
            bool: True if solution is valid, False otherwise
        """
        # Check probability properties
        if not (np.all(pi >= 0) and abs(np.sum(pi) - 1.0) < tol):
            return False
            
        # Check balance equations
        balance = np.dot(pi, A)
        if not np.all(np.abs(balance) < tol):
            return False
            
        return True
    
    def get_steady_state_distribution(self, pi: np.ndarray) -> Dict[int, float]:
        """Get distribution of number of busy units.
        
        Args:
            pi (numpy.ndarray): Steady-state probabilities
            
        Returns:
            Dict[int, float]: Distribution of number of busy units
        """
        distribution = {}
        for n in range(self.N + 1):
            prob = sum(p for i, p in enumerate(pi) if bin(i).count('1') == n)
            distribution[n] = prob
            
        return distribution