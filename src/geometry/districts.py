import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from .atoms import Point, Atom, AtomManager

@dataclass
class District:
    """Represents a service district."""
    id: int
    atoms: List[int]  # List of atom IDs in district
    center: Point  # District center (weighted by demand)
    total_demand: float = 0.0
    area: float = 0.0
    perimeter: float = 0.0
    compactness: float = 0.0  # Ratio of area to squared perimeter

class DistrictManager:
    """Manages district configuration and analysis."""
    
    def __init__(self, N: int, atom_manager: AtomManager):
        """Initialize district manager.
        
        Args:
            N (int): Number of districts
            atom_manager (AtomManager): Atom management system
        """
        self.N = N
        self.atom_manager = atom_manager
        self.districts = {}  # id -> District
        self.assignments = -np.ones(atom_manager.J, dtype=int)  # Atom -> District mapping
        
    def create_linear_districts(self) -> List[District]:
        """Create districts arranged linearly.
        
        Returns:
            List[District]: Created districts
        """
        atoms_per_district = self.atom_manager.J // self.N
        districts = []
        
        for i in range(self.N):
            start_idx = i * atoms_per_district
            end_idx = start_idx + atoms_per_district
            district_atoms = list(range(start_idx, end_idx))
            
            # Create district
            district = self._create_district(i, district_atoms)
            districts.append(district)
            self.districts[i] = district
            
            # Update assignments
            self.assignments[district_atoms] = i
            
        return districts
    
    def create_grid_districts(self, rows: int, cols: int) -> List[District]:
        """Create districts arranged in a grid.
        
        Args:
            rows (int): Number of district rows
            cols (int): Number of district columns
            
        Returns:
            List[District]: Created districts
        """
        if rows * cols != self.N:
            raise ValueError(f"Grid size {rows}x{cols} != number of districts {self.N}")
            
        atoms_per_row = self.atom_manager.J // (rows * cols)
        districts = []
        
        for i in range(self.N):
            district_row = i // cols
            district_col = i % cols
            
            # Calculate atom ranges for this district
            district_atoms = self._get_grid_district_atoms(
                district_row, district_col, rows, cols, atoms_per_row
            )
            
            # Create district
            district = self._create_district(i, district_atoms)
            districts.append(district)
            self.districts[i] = district
            
            # Update assignments
            self.assignments[district_atoms] = i
            
        return districts
    
    def _create_district(self, district_id: int, atom_ids: List[int]) -> District:
        """Create a district with given atoms.
        
        Args:
            district_id (int): District identifier
            atom_ids (List[int]): Atoms to include
            
        Returns:
            District: Created district
        """
        # Calculate district properties
        total_demand = sum(self.atom_manager.atoms[i].demand for i in atom_ids)
        area = sum(self.atom_manager.atoms[i].area for i in atom_ids)
        center = self.atom_manager.get_centroid(atom_ids)
        perimeter = self._calculate_perimeter(atom_ids)
        
        # Calculate compactness (normalized to [0,1])
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
        else:
            compactness = 0
            
        return District(
            id=district_id,
            atoms=atom_ids,
            center=center,
            total_demand=total_demand,
            area=area,
            perimeter=perimeter,
            compactness=compactness
        )
    
    def _calculate_perimeter(self, atom_ids: List[int]) -> float:
        """Calculate district perimeter.
        
        Args:
            atom_ids (List[int]): Atoms in district
            
        Returns:
            float: District perimeter
        """
        atom_set = set(atom_ids)
        perimeter = 0.0
        
        for atom_id in atom_ids:
            atom = self.atom_manager.atoms[atom_id]
            for neighbor in atom.neighbors:
                if neighbor not in atom_set:
                    # This edge is on the perimeter
                    edge_length = np.sqrt(
                        self.atom_manager.distances[atom_id, neighbor]
                    )
                    perimeter += edge_length
                    
        return perimeter
    
    def _get_grid_district_atoms(self, district_row: int, district_col: int,
                               rows: int, cols: int, atoms_per_row: int) -> List[int]:
        """Get atoms for district in grid arrangement."""
        atoms = []
        start_row = district_row * atoms_per_row
        start_col = district_col * atoms_per_row
        
        for r in range(atoms_per_row):
            for c in range(atoms_per_row):
                atom_row = start_row + r
                atom_col = start_col + c
                if (atom_row < self.atom_manager.J and 
                    atom_col < self.atom_manager.J):
                    atom_id = atom_row * cols * atoms_per_row + atom_col
                    atoms.append(atom_id)
                    
        return atoms
    
    def optimize_districts(self, max_iterations: int = 100, 
                         tolerance: float = 0.01) -> bool:
        """Optimize district assignments to balance workload.
        
        Args:
            max_iterations (int): Maximum optimization iterations
            tolerance (float): Convergence tolerance
            
        Returns:
            bool: True if optimization converged
        """
        for iteration in range(max_iterations):
            initial_imbalance = self.get_workload_imbalance()
            
            # Try to improve balance by moving boundary atoms
            improved = self._optimize_boundaries()
            
            # Check convergence
            if not improved:
                return True
                
            new_imbalance = self.get_workload_imbalance()
            if abs(new_imbalance - initial_imbalance) < tolerance:
                return True
                
        return False
    
    def _optimize_boundaries(self) -> bool:
        """Attempt to improve balance by moving boundary atoms."""
        improved = False
        
        # Get all boundary atoms
        boundary_atoms = self.get_boundary_atoms()
        
        for atom_id in boundary_atoms:
            current_district = self.assignments[atom_id]
            neighbors = self.atom_manager.atoms[atom_id].neighbors
            
            # Find neighboring districts
            neighbor_districts = set(
                self.assignments[n] for n in neighbors
                if self.assignments[n] != current_district
            )
            
            # Try moving atom to each neighboring district
            best_imbalance = self.get_workload_imbalance()
            best_district = current_district
            
            for district_id in neighbor_districts:
                # Temporarily move atom
                self.assignments[atom_id] = district_id
                self._update_districts([current_district, district_id])
                
                # Check if improvement
                new_imbalance = self.get_workload_imbalance()
                if new_imbalance < best_imbalance:
                    best_imbalance = new_imbalance
                    best_district = district_id
                    improved = True
                
                # Restore atom
                self.assignments[atom_id] = current_district
                self._update_districts([current_district, district_id])
            
            # Make best move permanent
            if best_district != current_district:
                self.assignments[atom_id] = best_district
                self._update_districts([current_district, best_district])
                
        return improved
    
    def _update_districts(self, district_ids: List[int]):
        """Update district properties after changes."""
        for district_id in district_ids:
            atom_ids = np.where(self.assignments == district_id)[0]
            self.districts[district_id] = self._create_district(
                district_id, atom_ids.tolist()
            )
            
    def get_workload_imbalance(self) -> float:
        """Calculate workload imbalance between districts.
        
        Returns:
            float: Workload imbalance measure
        """
        demands = [d.total_demand for d in self.districts.values()]
        return (max(demands) - min(demands)) / np.mean(demands)
    
    def get_boundary_atoms(self) -> Set[int]:
        """Get atoms on district boundaries.
        
        Returns:
            Set[int]: Boundary atom IDs
        """
        boundary = set()
        
        for atom_id in range(self.atom_manager.J):
            district = self.assignments[atom_id]
            neighbors = self.atom_manager.atoms[atom_id].neighbors
            
            # Check if any neighbors are in different districts
            if any(self.assignments[n] != district for n in neighbors):
                boundary.add(atom_id)
                
        return boundary
    
    def get_district_metrics(self) -> Dict:
        """Get comprehensive district metrics.
        
        Returns:
            Dict: District metrics
        """
        return {
            'workload_imbalance': self.get_workload_imbalance(),
            'average_compactness': np.mean([
                d.compactness for d in self.districts.values()
            ]),
            'boundary_fraction': len(self.get_boundary_atoms()) / self.atom_manager.J,
            'district_sizes': {
                i: len(d.atoms) for i, d in self.districts.items()
            },
            'district_demands': {
                i: d.total_demand for i, d in self.districts.items()
            },
            'district_areas': {
                i: d.area for i, d in self.districts.items()
            }
        }