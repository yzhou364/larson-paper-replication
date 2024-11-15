"""
Optimization components for hypercube queuing model:
- District boundary optimization
- Unit location optimization
- Dispatch policy optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, differential_evolution
import copy

@dataclass
class OptimizationConfig:
    """Configuration for optimization procedures."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    population_size: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    use_constraints: bool = True
    multi_objective: bool = False

class BaseOptimizer:
    """Base class for optimization implementations."""
    
    def __init__(self, model: 'HypercubeModel', config: Optional[OptimizationConfig] = None):
        """Initialize optimizer.
        
        Args:
            model (HypercubeModel): Model to optimize
            config (Optional[OptimizationConfig]): Optimization configuration
        """
        self.model = model
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        self.history = []
        
    def _record_iteration(self, iteration: int, solution: np.ndarray, 
                         objective: float):
        """Record optimization iteration."""
        self.history.append({
            'iteration': iteration,
            'solution': solution.copy(),
            'objective': objective
        })
        
    def get_optimization_summary(self) -> Dict:
        """Get optimization progress summary."""
        return {
            'iterations': len(self.history),
            'initial_objective': self.history[0]['objective'],
            'final_objective': self.history[-1]['objective'],
            'improvement': (self.history[0]['objective'] - 
                          self.history[-1]['objective']) / self.history[0]['objective'],
            'convergence': self._check_convergence()
        }
        
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.history) < 2:
            return False
            
        recent_objectives = [h['objective'] for h in self.history[-10:]]
        return np.std(recent_objectives) < self.config.tolerance

class DistrictOptimizer(BaseOptimizer):
    """Optimizes district boundaries."""
    
    def optimize(self) -> Dict:
        """Run district optimization.
        
        Returns:
            Dict: Optimization results
        """
        initial_districts = copy.deepcopy(self.model.district_manager.districts)
        best_solution = initial_districts
        best_objective = self._evaluate_districts(initial_districts)
        
        self._record_iteration(0, self._encode_districts(initial_districts), best_objective)
        
        for iteration in range(self.config.max_iterations):
            # Generate candidate solutions
            candidates = self._generate_district_candidates(best_solution)
            
            # Evaluate candidates
            for candidate in candidates:
                objective = self._evaluate_districts(candidate)
                if objective < best_objective:
                    best_solution = candidate
                    best_objective = objective
                    
            self._record_iteration(
                iteration + 1,
                self._encode_districts(best_solution),
                best_objective
            )
            
            if self._check_convergence():
                break
                
        return {
            'initial_districts': initial_districts,
            'optimized_districts': best_solution,
            'history': self.history,
            'summary': self.get_optimization_summary()
        }
        
    def _evaluate_districts(self, districts: List[List[int]]) -> float:
        """Evaluate district configuration."""
        # Apply districts to model
        self.model.district_manager.set_districts(districts)
        results = self.model.run()
        
        # Compute weighted objective
        return (
            0.4 * results['workload_imbalance'] +
            0.4 * results['mean_travel_time'] +
            0.2 * results['interdistrict_fraction']
        )
        
    def _generate_district_candidates(self, current: List[List[int]]) -> List[List[List[int]]]:
        """Generate candidate district configurations."""
        candidates = []
        
        # Move boundary atoms
        boundary_atoms = self.model.district_manager.get_boundary_atoms()
        for atom in boundary_atoms:
            current_district = self._find_district(atom, current)
            neighbors = self.model.atom_manager.get_atom_neighbors(atom)
            
            for neighbor in neighbors:
                neighbor_district = self._find_district(neighbor, current)
                if neighbor_district != current_district:
                    # Create new configuration moving atom to neighbor district
                    new_districts = copy.deepcopy(current)
                    new_districts[current_district].remove(atom)
                    new_districts[neighbor_district].append(atom)
                    candidates.append(new_districts)
                    
        return candidates
    
    def _find_district(self, atom: int, districts: List[List[int]]) -> int:
        """Find district containing atom."""
        for i, district in enumerate(districts):
            if atom in district:
                return i
        return -1
    
    def _encode_districts(self, districts: List[List[int]]) -> np.ndarray:
        """Encode districts for history recording."""
        encoding = np.zeros(self.model.J)
        for i, district in enumerate(districts):
            encoding[district] = i
        return encoding

class LocationOptimizer(BaseOptimizer):
    """Optimizes unit locations."""
    
    def optimize(self) -> Dict:
        """Run location optimization.
        
        Returns:
            Dict: Optimization results
        """
        # Initial locations
        initial_locations = self.model.get_unit_locations()
        
        # Define bounds
        bounds = [(0, 1) for _ in range(len(initial_locations))]
        
        # Run differential evolution
        result = differential_evolution(
            self._evaluate_locations,
            bounds,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size,
            mutation=self.config.mutation_rate,
            recombination=self.config.crossover_rate,
            callback=self._optimization_callback
        )
        
        optimized_locations = result.x.reshape(-1, 2)
        
        return {
            'initial_locations': initial_locations,
            'optimized_locations': optimized_locations,
            'history': self.history,
            'summary': self.get_optimization_summary()
        }
        
    def _evaluate_locations(self, locations: np.ndarray) -> float:
        """Evaluate unit locations."""
        locations = locations.reshape(-1, 2)
        self.model.set_unit_locations(locations)
        results = self.model.run()
        
        return results['mean_travel_time']
    
    def _optimization_callback(self, xk: np.ndarray, convergence: float):
        """Callback for optimization progress."""
        objective = self._evaluate_locations(xk)
        self._record_iteration(len(self.history), xk, objective)

class DispatchOptimizer(BaseOptimizer):
    """Optimizes dispatch policy parameters."""
    
    def optimize(self) -> Dict:
        """Run dispatch policy optimization.
        
        Returns:
            Dict: Optimization results
        """
        initial_policy = self.model.get_dispatch_policy()
        
        if self.config.multi_objective:
            return self._multi_objective_optimization(initial_policy)
        else:
            return self._single_objective_optimization(initial_policy)
    
    def _single_objective_optimization(self, initial_policy: Dict) -> Dict:
        """Run single-objective optimization."""
        # Define bounds for policy parameters
        bounds = self._get_policy_bounds()
        
        # Run optimization
        result = minimize(
            self._evaluate_policy,
            self._encode_policy(initial_policy),
            method='SLSQP',
            bounds=bounds,
            constraints=self._get_constraints() if self.config.use_constraints else None,
            options={'maxiter': self.config.max_iterations}
        )
        
        optimized_policy = self._decode_policy(result.x)
        
        return {
            'initial_policy': initial_policy,
            'optimized_policy': optimized_policy,
            'history': self.history,
            'summary': self.get_optimization_summary()
        }
        
    def _multi_objective_optimization(self, initial_policy: Dict) -> Dict:
        """Run multi-objective optimization."""
        population = self._initialize_population(initial_policy)
        pareto_front = []
        
        for iteration in range(self.config.max_iterations):
            # Evaluate population
            objectives = [self._evaluate_policy_multi(p) for p in population]
            
            # Update Pareto front
            pareto_front = self._update_pareto_front(population, objectives, pareto_front)
            
            # Generate new population
            population = self._evolve_population(population, objectives)
            
            self._record_iteration(
                iteration,
                self._encode_policy(population[0]),
                objectives[0][0]  # Record first objective
            )
            
            if self._check_convergence():
                break
                
        return {
            'initial_policy': initial_policy,
            'pareto_front': pareto_front,
            'history': self.history,
            'summary': self.get_optimization_summary()
        }
        
    def _evaluate_policy(self, encoded_policy: np.ndarray) -> float:
        """Evaluate single policy configuration."""
        policy = self._decode_policy(encoded_policy)
        self.model.set_dispatch_policy(policy)
        results = self.model.run()
        
        return (
            0.4 * results['mean_travel_time'] +
            0.3 * results['workload_imbalance'] +
            0.3 * results['interdistrict_fraction']
        )
        
    def _evaluate_policy_multi(self, policy: Dict) -> Tuple[float, float, float]:
        """Evaluate policy for multiple objectives."""
        self.model.set_dispatch_policy(policy)
        results = self.model.run()
        
        return (
            results['mean_travel_time'],
            results['workload_imbalance'],
            results['interdistrict_fraction']
        )
        
    def _get_policy_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for policy parameters."""
        return [
            (0, 1),  # Weight bounds
            (0, 10),  # Time threshold bounds
            (0, 1)   # Priority factor bounds
        ]
        
    def _get_constraints(self) -> List[Dict]:
        """Get optimization constraints."""
        return [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:3]) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negativity
        ]
        
    def _encode_policy(self, policy: Dict) -> np.ndarray:
        """Encode policy for optimization."""
        return np.array([
            policy['weights']['travel_time'],
            policy['weights']['workload'],
            policy['weights']['district'],
            policy['time_threshold'],
            policy['priority_factor']
        ])
        
    def _decode_policy(self, encoded: np.ndarray) -> Dict:
        """Decode optimization result to policy."""
        return {
            'weights': {
                'travel_time': encoded[0],
                'workload': encoded[1],
                'district': encoded[2]
            },
            'time_threshold': encoded[3],
            'priority_factor': encoded[4]
        }
        
    def _initialize_population(self, initial_policy: Dict) -> List[Dict]:
        """Initialize population for multi-objective optimization."""
        population = [initial_policy]
        for _ in range(self.config.population_size - 1):
            # Generate random variations
            new_policy = copy.deepcopy(initial_policy)
            for param in new_policy['weights'].values():
                param *= np.random.uniform(0.8, 1.2)
            new_policy['time_threshold'] *= np.random.uniform(0.8, 1.2)
            new_policy['priority_factor'] *= np.random.uniform(0.8, 1.2)
            population.append(new_policy)
            
        return population
        
    def _update_pareto_front(self, population: List[Dict],
                           objectives: List[Tuple[float, float, float]],
                           pareto_front: List[Dict]) -> List[Dict]:
        """Update Pareto front with new solutions."""
        for policy, obj in zip(population, objectives):
            dominated = False
            for front_obj in [self._evaluate_policy_multi(p) for p in pareto_front]:
                if all(o1 <= o2 for o1, o2 in zip(front_obj, obj)):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(copy.deepcopy(policy))
                
        # Remove dominated solutions from front
        i = 0
        while i < len(pareto_front):
            obj_i = self._evaluate_policy_multi(pareto_front[i])
            dominated = False
            j = i + 1
            while j < len(pareto_front):
                obj_j = self._evaluate_policy_multi(pareto_front[j])
                if all(o1 <= o2 for o1, o2 in zip(obj_j, obj_i)):
                    dominated = True
                    break
                j += 1
            if dominated:
                pareto_front.pop(i)
            else:
                i += 1
                
        return pareto_front
        
    def _evolve_population(self, population: List[Dict],
                         objectives: List[Tuple[float, float, float]]) -> List[Dict]:
        """Evolve population for next generation."""
        new_population = []
        
        # Elite preservation
        sorted_indices = np.argsort([sum(obj) for obj in objectives])
        elite_size = max(1, self.config.population_size // 10)
        new_population.extend([population[i] for i in sorted_indices[:elite_size]])
        
        # Generate remaining through crossover and mutation
        while len(new_population) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate:
                # Crossover
                parent1, parent2 = np.random.choice(population, 2)
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = np.random.choice(population)
                child = self._mutate(parent)
                
            new_population.append(child)
            
        return new_population
        
    def _crossover(self, policy1: Dict, policy2: Dict) -> Dict:
        """Perform crossover between policies."""
        child = copy.deepcopy(policy1)
        
        # Crossover weights
        for key in child['weights']:
            if np.random.random() < 0.5:
                child['weights'][key] = policy2['weights'][key]
                
        # Crossover other parameters
        if np.random.random() < 0.5:
            child['time_threshold'] = policy2['time_threshold']
        if np.random.random() < 0.5:
            child['priority_factor'] = policy2['priority_factor']
            
        return child
        
    def _mutate(self, policy: Dict) -> Dict:
        """Mutate policy parameters."""
        mutated = copy.deepcopy(policy)
        
        # Mutate weights
        for key in mutated['weights']:
            if np.random.random() < self.config.mutation_rate:
                mutated['weights'][key] *= np.random.uniform(0.8, 1.2)
                
        # Mutate time threshold
        if np.random.random() < self.config.mutation_rate:
            mutated['time_threshold'] *= np.random.uniform(0.8, 1.2)
            
        # Mutate priority factor
        if np.random.random() < self.config.mutation_rate:
            mutated['priority_factor'] *= np.random.uniform(0.8, 1.2)
            
        # Normalize weights
        weight_sum = sum(mutated['weights'].values())
        for key in mutated['weights']:
            mutated['weights'][key] /= weight_sum
            
        return mutated
        
    def get_policy_recommendations(self) -> Dict:
        """Generate policy recommendations based on optimization results.
        
        Returns:
            Dict: Policy recommendations and explanations
        """
        if not self.history:
            return {}
            
        initial_objective = self.history[0]['objective']
        final_objective = self.history[-1]['objective']
        improvement = (initial_objective - final_objective) / initial_objective
        
        recommendations = {
            'improvement_summary': {
                'percentage': improvement * 100,
                'significant': improvement > 0.1
            },
            'parameter_changes': self._analyze_parameter_changes(),
            'trade_offs': self._analyze_trade_offs(),
            'implementation': self._get_implementation_guidelines()
        }
        
        return recommendations
        
    def _analyze_parameter_changes(self) -> Dict:
        """Analyze how parameters changed during optimization."""
        initial_policy = self._decode_policy(self.history[0]['solution'])
        final_policy = self._decode_policy(self.history[-1]['solution'])
        
        changes = {}
        
        # Analyze weight changes
        for key in initial_policy['weights']:
            initial = initial_policy['weights'][key]
            final = final_policy['weights'][key]
            changes[f'weight_{key}'] = {
                'initial': initial,
                'final': final,
                'change': (final - initial) / initial,
                'significant': abs(final - initial) / initial > 0.1
            }
            
        # Analyze other parameter changes
        for param in ['time_threshold', 'priority_factor']:
            initial = initial_policy[param]
            final = final_policy[param]
            changes[param] = {
                'initial': initial,
                'final': final,
                'change': (final - initial) / initial,
                'significant': abs(final - initial) / initial > 0.1
            }
            
        return changes
        
    def _analyze_trade_offs(self) -> Dict:
        """Analyze trade-offs in optimization results."""
        if not self.config.multi_objective:
            return {}
            
        trade_offs = {
            'conflicts': [],
            'synergies': [],
            'key_compromises': []
        }
        
        # Analyze Pareto front if available
        if hasattr(self, 'pareto_front'):
            objectives = [self._evaluate_policy_multi(p) for p in self.pareto_front]
            
            # Find conflicting objectives
            correlations = np.corrcoef(np.array(objectives).T)
            for i in range(3):
                for j in range(i+1, 3):
                    if correlations[i,j] < -0.5:
                        trade_offs['conflicts'].append({
                            'objective1': ['travel_time', 'workload', 'interdistrict'][i],
                            'objective2': ['travel_time', 'workload', 'interdistrict'][j],
                            'correlation': correlations[i,j]
                        })
                    elif correlations[i,j] > 0.5:
                        trade_offs['synergies'].append({
                            'objective1': ['travel_time', 'workload', 'interdistrict'][i],
                            'objective2': ['travel_time', 'workload', 'interdistrict'][j],
                            'correlation': correlations[i,j]
                        })
                        
            # Identify key compromises
            for policy in self.pareto_front:
                objs = self._evaluate_policy_multi(policy)
                if all(o <= np.median([obj[i] for obj in objectives]) 
                      for i, o in enumerate(objs)):
                    trade_offs['key_compromises'].append({
                        'policy': policy,
                        'objectives': objs
                    })
                    
        return trade_offs
        
    def _get_implementation_guidelines(self) -> Dict:
        """Generate guidelines for implementing optimized policy."""
        final_policy = self._decode_policy(self.history[-1]['solution'])
        
        guidelines = {
            'critical_parameters': [],
            'stability_concerns': [],
            'implementation_steps': []
        }
        
        # Identify critical parameters
        for key, value in final_policy['weights'].items():
            if value > 0.4:  # Parameter has high weight
                guidelines['critical_parameters'].append({
                    'parameter': key,
                    'importance': 'high',
                    'value': value,
                    'notes': f"Careful tuning required for {key}"
                })
                
        # Identify stability concerns
        recent_objectives = [h['objective'] for h in self.history[-10:]]
        if np.std(recent_objectives) > self.config.tolerance:
            guidelines['stability_concerns'].append(
                "Policy shows sensitivity to parameter changes"
            )
            
        # Generate implementation steps
        guidelines['implementation_steps'] = [
            {
                'step': 1,
                'action': "Update dispatch weights",
                'details': final_policy['weights']
            },
            {
                'step': 2,
                'action': "Set time threshold",
                'value': final_policy['time_threshold']
            },
            {
                'step': 3,
                'action': "Configure priority factor",
                'value': final_policy['priority_factor']
            },
            {
                'step': 4,
                'action': "Monitor performance metrics",
                'metrics': ['travel_time', 'workload_balance', 'interdistrict_fraction']
            }
        ]
        
        return guidelines