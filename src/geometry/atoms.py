import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Point:
    """2D point representation."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def manhattan_distance_to(self, other: 'Point') -> float:
        """Calculate Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y)

@dataclass
class Atom:
    """Geographical atom representation."""
    id: int
    center: Point
    demand: float = 0.0  # Demand rate for this atom
    area: float = 1.0    # Area of the atom
    neighbors: List[int] = None  # IDs of neighboring atoms
    
    def __post_init__(self):
        """Initialize neighbors list if not provided."""
        if self.neighbors is None:
            self.neighbors = []

class AtomManager:
    """Manages geographical atoms in the system."""
    
    def __init__(self, J: int):
        """Initialize atom manager.
        
        Args:
            J (int): Number of atoms
        """
        self.J = J
        self.atoms = {}  # id -> Atom
        self.demands = np.zeros(J)  # Demand rates
        self.distances = np.zeros((J, J))  # Distance matrix
        
    def create_linear_atoms(self, length: float) -> List[Atom]:
        """Create atoms arranged in a line.
        
        Args:
            length (float): Total length of the line
            
        Returns:
            List[Atom]: Created atoms
        """
        atom_length = length / self.J
        atoms = []
        
        for i in range(self.J):
            center = Point(
                x=i * atom_length + atom_length/2,
                y=0.0
            )
            atom = Atom(
                id=i,
                center=center,
                area=atom_length,
                neighbors=self._get_linear_neighbors(i)
            )
            atoms.append(atom)
            self.atoms[i] = atom
            
        self._update_distance_matrix()
        return atoms
    
    def create_grid_atoms(self, width: float, height: float, 
                         rows: int, cols: int) -> List[Atom]:
        """Create atoms arranged in a grid.
        
        Args:
            width (float): Grid width
            height (float): Grid height
            rows (int): Number of rows
            cols (int): Number of columns
            
        Returns:
            List[Atom]: Created atoms
        """
        if rows * cols != self.J:
            raise ValueError(f"Grid size {rows}x{cols} != number of atoms {self.J}")
            
        cell_width = width / cols
        cell_height = height / rows
        atoms = []
        
        for i in range(rows):
            for j in range(cols):
                atom_id = i * cols + j
                center = Point(
                    x=j * cell_width + cell_width/2,
                    y=i * cell_height + cell_height/2
                )
                atom = Atom(
                    id=atom_id,
                    center=center,
                    area=cell_width * cell_height,
                    neighbors=self._get_grid_neighbors(i, j, rows, cols)
                )
                atoms.append(atom)
                self.atoms[atom_id] = atom
                
        self._update_distance_matrix()
        return atoms
    
    def set_demands(self, demands: np.ndarray):
        """Set demand rates for atoms.
        
        Args:
            demands (numpy.ndarray): Demand rates
        """
        if len(demands) != self.J:
            raise ValueError(f"Number of demands {len(demands)} != number of atoms {self.J}")
            
        self.demands = demands.copy()
        for i, demand in enumerate(demands):
            if i in self.atoms:
                self.atoms[i].demand = demand
                
    def get_atom_distances(self, method: str = 'euclidean') -> np.ndarray:
        """Get distance matrix between atoms.
        
        Args:
            method (str): Distance calculation method ('euclidean' or 'manhattan')
            
        Returns:
            numpy.ndarray: Distance matrix
        """
        distances = np.zeros((self.J, self.J))
        
        for i in range(self.J):
            for j in range(self.J):
                if i == j:
                    # Intra-atom distance (use area-based approximation)
                    distances[i,j] = np.sqrt(self.atoms[i].area) / 2
                else:
                    if method == 'euclidean':
                        distances[i,j] = self.atoms[i].center.distance_to(
                            self.atoms[j].center
                        )
                    else:  # manhattan
                        distances[i,j] = self.atoms[i].center.manhattan_distance_to(
                            self.atoms[j].center
                        )
                        
        return distances
    
    def _get_linear_neighbors(self, i: int) -> List[int]:
        """Get neighbor indices for linear arrangement."""
        neighbors = []
        if i > 0:
            neighbors.append(i-1)
        if i < self.J - 1:
            neighbors.append(i+1)
        return neighbors
    
    def _get_grid_neighbors(self, row: int, col: int, 
                          rows: int, cols: int) -> List[int]:
        """Get neighbor indices for grid arrangement."""
        neighbors = []
        # Check all adjacent cells (including diagonals)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if (0 <= new_row < rows and 
                    0 <= new_col < cols):
                    neighbors.append(new_row * cols + new_col)
        return neighbors
    
    def _update_distance_matrix(self):
        """Update internal distance matrix."""
        self.distances = self.get_atom_distances()
    
    def get_atom_coverage(self, point: Point, radius: float) -> List[int]:
        """Get atoms within coverage radius of point.
        
        Args:
            point (Point): Center point
            radius (float): Coverage radius
            
        Returns:
            List[int]: IDs of covered atoms
        """
        covered = []
        for atom_id, atom in self.atoms.items():
            if atom.center.distance_to(point) <= radius:
                covered.append(atom_id)
        return covered
    
    def get_centroid(self, atom_ids: List[int]) -> Optional[Point]:
        """Calculate centroid of specified atoms.
        
        Args:
            atom_ids (List[int]): Atoms to consider
            
        Returns:
            Optional[Point]: Centroid point
        """
        if not atom_ids:
            return None
            
        # Weight by demand
        total_demand = sum(self.atoms[i].demand for i in atom_ids)
        if total_demand == 0:
            # Unweighted centroid
            x = sum(self.atoms[i].center.x for i in atom_ids) / len(atom_ids)
            y = sum(self.atoms[i].center.y for i in atom_ids) / len(atom_ids)
        else:
            # Demand-weighted centroid
            x = sum(self.atoms[i].center.x * self.atoms[i].demand 
                   for i in atom_ids) / total_demand
            y = sum(self.atoms[i].center.y * self.atoms[i].demand 
                   for i in atom_ids) / total_demand
            
        return Point(x, y)
    
    def get_atom_stats(self) -> Dict:
        """Get statistics about atoms.
        
        Returns:
            Dict: Atom statistics
        """
        return {
            'total_atoms': self.J,
            'total_demand': np.sum(self.demands),
            'max_demand': np.max(self.demands),
            'min_demand': np.min(self.demands),
            'average_area': np.mean([atom.area for atom in self.atoms.values()]),
            'average_neighbors': np.mean([len(atom.neighbors) 
                                       for atom in self.atoms.values()]),
            'max_distance': np.max(self.distances),
            'average_distance': np.mean(self.distances)
        }