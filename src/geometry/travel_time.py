import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .atoms import Point, AtomManager
from .districts import DistrictManager

@dataclass
class TravelTimeConfig:
    """Configuration for travel time calculations."""
    base_speed: float = 1.0  # Base travel speed
    congestion_factor: float = 1.2  # Multiplier for congested areas
    peak_hours: List[int] = None  # Hours considered peak (0-23)
    time_penalties: Dict[str, float] = None  # Additional time penalties
    
    def __post_init__(self):
        """Initialize default values."""
        if self.peak_hours is None:
            self.peak_hours = [7, 8, 9, 16, 17, 18]  # Default peak hours
        if self.time_penalties is None:
            self.time_penalties = {
                'intersection': 0.1,  # Time penalty for crossing intersections
                'turn': 0.2,         # Time penalty for turns
                'boundary': 0.15     # Time penalty for crossing district boundaries
            }

class TravelTimeCalculator:
    """Calculates travel times between atoms and districts."""
    
    def __init__(self, atom_manager: AtomManager, 
                 district_manager: Optional[DistrictManager] = None,
                 config: Optional[TravelTimeConfig] = None):
        """Initialize travel time calculator.
        
        Args:
            atom_manager (AtomManager): Atom management system
            district_manager (Optional[DistrictManager]): District management system
            config (Optional[TravelTimeConfig]): Travel time configuration
        """
        self.atom_manager = atom_manager
        self.district_manager = district_manager
        self.config = config or TravelTimeConfig()
        
        # Initialize matrices
        self.travel_times = np.zeros((atom_manager.J, atom_manager.J))
        self.peak_times = np.zeros((atom_manager.J, atom_manager.J))
        self.distance_matrix = atom_manager.get_atom_distances()
        
        # Calculate base travel times
        self._calculate_base_travel_times()
        
    def _calculate_base_travel_times(self):
        """Calculate base travel times between all atoms."""
        # Convert distances to times using base speed
        self.travel_times = self.distance_matrix / self.config.base_speed
        
        # Add intersection penalties
        self._add_intersection_penalties()
        
        # Add turn penalties
        self._add_turn_penalties()
        
        # Add boundary crossing penalties if districts are defined
        if self.district_manager is not None:
            self._add_boundary_penalties()
            
        # Calculate peak hour times
        self.peak_times = self.travel_times * self.config.congestion_factor
        
    def _add_intersection_penalties(self):
        """Add time penalties for crossing intersections."""
        for i in range(self.atom_manager.J):
            for j in range(self.atom_manager.J):
                if i != j:
                    # Count intersections along path
                    source = self.atom_manager.atoms[i]
                    dest = self.atom_manager.atoms[j]
                    intersections = self._count_intersections(source, dest)
                    
                    # Add penalties
                    self.travel_times[i,j] += (
                        intersections * self.config.time_penalties['intersection']
                    )
                    
    def _add_turn_penalties(self):
        """Add time penalties for turns."""
        for i in range(self.atom_manager.J):
            for j in range(self.atom_manager.J):
                if i != j:
                    # Count turns along path
                    source = self.atom_manager.atoms[i]
                    dest = self.atom_manager.atoms[j]
                    turns = self._count_turns(source, dest)
                    
                    # Add penalties
                    self.travel_times[i,j] += (
                        turns * self.config.time_penalties['turn']
                    )
                    
    def _add_boundary_penalties(self):
        """Add time penalties for crossing district boundaries."""
        if self.district_manager is None:
            return
            
        for i in range(self.atom_manager.J):
            for j in range(self.atom_manager.J):
                if i != j:
                    # Check if path crosses district boundaries
                    source_district = self.district_manager.assignments[i]
                    dest_district = self.district_manager.assignments[j]
                    
                    if source_district != dest_district:
                        # Add boundary crossing penalty
                        self.travel_times[i,j] += self.config.time_penalties['boundary']
                        
    def _count_intersections(self, source: 'Atom', dest: 'Atom') -> int:
        """Count number of intersections along path.
        
        Args:
            source (Atom): Starting atom
            dest (Atom): Destination atom
            
        Returns:
            int: Number of intersections
        """
        # Simple approximation based on Manhattan distance
        dx = abs(source.center.x - dest.center.x)
        dy = abs(source.center.y - dest.center.y)
        return int(dx + dy)  # Each grid line crossing is an intersection
    
    def _count_turns(self, source: 'Atom', dest: 'Atom') -> int:
        """Count number of turns along path.
        
        Args:
            source (Atom): Starting atom
            dest (Atom): Destination atom
            
        Returns:
            int: Number of turns
        """
        # Simple approximation based on relative positions
        if source.center.x == dest.center.x or source.center.y == dest.center.y:
            return 0  # Straight path
        else:
            return 1  # One turn needed
    
    def get_travel_time(self, from_atom: int, to_atom: int, 
                       hour: Optional[int] = None) -> float:
        """Get travel time between atoms.
        
        Args:
            from_atom (int): Source atom ID
            to_atom (int): Destination atom ID
            hour (Optional[int]): Hour of day (0-23)
            
        Returns:
            float: Travel time
        """
        if hour is not None and hour in self.config.peak_hours:
            return self.peak_times[from_atom, to_atom]
        return self.travel_times[from_atom, to_atom]
    
    def get_district_travel_times(self) -> np.ndarray:
        """Calculate average travel times between districts.
        
        Returns:
            numpy.ndarray: District travel time matrix
        """
        if self.district_manager is None:
            raise ValueError("District manager not initialized")
            
        N = self.district_manager.N
        district_times = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                # Get atoms in each district
                dist_i_atoms = self.district_manager.districts[i].atoms
                dist_j_atoms = self.district_manager.districts[j].atoms
                
                # Calculate average travel time
                times = []
                for from_atom in dist_i_atoms:
                    for to_atom in dist_j_atoms:
                        times.append(self.travel_times[from_atom, to_atom])
                        
                district_times[i,j] = np.mean(times)
                
        return district_times
    
    def analyze_coverage(self, time_threshold: float) -> Dict:
        """Analyze coverage within time threshold.
        
        Args:
            time_threshold (float): Maximum acceptable travel time
            
        Returns:
            Dict: Coverage analysis
        """
        coverage = {}
        
        # Analyze atom coverage
        atom_coverage = np.zeros(self.atom_manager.J)
        for i in range(self.atom_manager.J):
            covered_atoms = np.where(self.travel_times[i,:] <= time_threshold)[0]
            atom_coverage[i] = len(covered_atoms) / self.atom_manager.J
            
        coverage['atom_coverage'] = {
            'mean': np.mean(atom_coverage),
            'min': np.min(atom_coverage),
            'max': np.max(atom_coverage)
        }
        
        # Analyze district coverage if available
        if self.district_manager is not None:
            district_times = self.get_district_travel_times()
            district_coverage = np.zeros(self.district_manager.N)
            
            for i in range(self.district_manager.N):
                covered_districts = np.where(district_times[i,:] <= time_threshold)[0]
                district_coverage[i] = len(covered_districts) / self.district_manager.N
                
            coverage['district_coverage'] = {
                'mean': np.mean(district_coverage),
                'min': np.min(district_coverage),
                'max': np.max(district_coverage)
            }
            
        return coverage
    
    def get_travel_time_stats(self) -> Dict:
        """Get travel time statistics.
        
        Returns:
            Dict: Travel time statistics
        """
        return {
            'min_time': np.min(self.travel_times[self.travel_times > 0]),
            'max_time': np.max(self.travel_times),
            'mean_time': np.mean(self.travel_times),
            'std_time': np.std(self.travel_times),
            'peak_multiplier': np.mean(self.peak_times / self.travel_times),
            'boundary_effect': self._calculate_boundary_effect()
        }
    
    def _calculate_boundary_effect(self) -> float:
        """Calculate effect of boundary crossings on travel times."""
        if self.district_manager is None:
            return 0.0
            
        # Compare intra-district to inter-district times
        intra_times = []
        inter_times = []
        
        for i in range(self.atom_manager.J):
            for j in range(self.atom_manager.J):
                if i != j:
                    time = self.travel_times[i,j]
                    if (self.district_manager.assignments[i] == 
                        self.district_manager.assignments[j]):
                        intra_times.append(time)
                    else:
                        inter_times.append(time)
                        
        if not intra_times or not inter_times:
            return 0.0
            
        return np.mean(inter_times) / np.mean(intra_times) - 1.0