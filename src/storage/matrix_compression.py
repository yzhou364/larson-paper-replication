import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class CompressedElement:
    """Represents a compressed matrix element."""
    row: int
    col: int
    value: float
    state_weight: int  # Number of ones in binary representation of state

class MatrixCompressor:
    """Efficient storage and manipulation of hypercube transition matrices."""
    
    def __init__(self, N: int):
        """Initialize matrix compressor.
        
        Args:
            N (int): Number of units/dimensions
        """
        self.N = N
        self.size = 2**N
        self.map_array = np.zeros(self.size, dtype=int)
        self.compressed_data = defaultdict(dict)
        self.initialize_mapping()
        
    def initialize_mapping(self):
        """Initialize MAP array for compressed storage."""
        current_index = 0
        for j in range(self.size):
            self.map_array[j] = current_index
            state_weight = bin(j).count('1')  # Number of ones
            current_index += state_weight + 1
            
    def compress_matrix(self, A: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Compress full matrix into efficient storage format.
        
        Args:
            A (numpy.ndarray): Full transition matrix
            
        Returns:
            Tuple[Dict, numpy.ndarray]: Compressed data and mapping array
        """
        if A.shape != (self.size, self.size):
            raise ValueError(f"Matrix must be {self.size}x{self.size}")
            
        self.compressed_data.clear()
        
        for i in range(self.size):
            state_i = format(i, f'0{self.N}b')
            weight_i = state_i.count('1')
            
            for j in range(self.size):
                if abs(A[i,j]) > 1e-10:  # Store only non-zero elements
                    map_index = self.map_array[i]
                    self.compressed_data[map_index + weight_i][j] = A[i,j]
                    
        return self.compressed_data, self.map_array
    
    def decompress_matrix(self) -> np.ndarray:
        """Reconstruct full matrix from compressed format.
        
        Returns:
            numpy.ndarray: Full transition matrix
        """
        A = np.zeros((self.size, self.size))
        
        for i in range(self.size):
            map_index = self.map_array[i]
            state_weight = format(i, f'0{self.N}b').count('1')
            
            for k in range(state_weight + 1):
                if map_index + k in self.compressed_data:
                    for j, value in self.compressed_data[map_index + k].items():
                        A[i,j] = value
                        
        return A
    
    def store_element(self, from_state: int, to_state: int, 
                     value: float) -> bool:
        """Store single matrix element in compressed format.
        
        Args:
            from_state (int): Source state
            to_state (int): Destination state
            value (float): Transition rate
            
        Returns:
            bool: True if stored successfully
        """
        if abs(value) <= 1e-10:
            return False
            
        map_index = self.map_array[from_state]
        state_weight = format(from_state, f'0{self.N}b').count('1')
        self.compressed_data[map_index + state_weight][to_state] = value
        return True
    
    def get_element(self, from_state: int, to_state: int) -> float:
        """Retrieve matrix element from compressed storage.
        
        Args:
            from_state (int): Source state
            to_state (int): Destination state
            
        Returns:
            float: Stored value (0.0 if not found)
        """
        map_index = self.map_array[from_state]
        state_weight = format(from_state, f'0{self.N}b').count('1')
        return self.compressed_data[map_index + state_weight].get(to_state, 0.0)
    
    def get_row(self, state: int) -> Dict[int, float]:
        """Get all non-zero elements in specified row.
        
        Args:
            state (int): State number (row index)
            
        Returns:
            Dict[int, float]: Dictionary of column indices to values
        """
        map_index = self.map_array[state]
        state_weight = format(state, f'0{self.N}b').count('1')
        return dict(self.compressed_data[map_index + state_weight])
    
    def update_row(self, state: int, values: Dict[int, float]):
        """Update multiple elements in specified row.
        
        Args:
            state (int): State number (row index)
            values (Dict[int, float]): Dictionary of column indices to values
        """
        map_index = self.map_array[state]
        state_weight = format(state, f'0{self.N}b').count('1')
        
        # Filter out near-zero values
        filtered_values = {
            col: val for col, val in values.items() 
            if abs(val) > 1e-10
        }
        
        if filtered_values:
            self.compressed_data[map_index + state_weight].update(filtered_values)
        else:
            # Remove empty entries
            self.compressed_data.pop(map_index + state_weight, None)
            
    def get_matrix_stats(self) -> Dict:
        """Compute statistics about matrix compression.
        
        Returns:
            Dict: Compression statistics
        """
        total_elements = self.size * self.size
        stored_elements = sum(len(row) for row in self.compressed_data.values())
        
        return {
            'matrix_size': self.size,
            'total_elements': total_elements,
            'stored_elements': stored_elements,
            'compression_ratio': 1 - (stored_elements / total_elements),
            'density': stored_elements / total_elements,
            'memory_saved': 1 - (
                (stored_elements * 3 + len(self.map_array)) / 
                (total_elements)
            )
        }
    
    def validate_compression(self, original: Optional[np.ndarray] = None) -> bool:
        """Validate compressed storage.
        
        Args:
            original (Optional[numpy.ndarray]): Original matrix for comparison
            
        Returns:
            bool: True if validation passes
        """
        # Check map array
        if len(self.map_array) != self.size:
            return False
            
        # Check that map array is strictly increasing
        if not all(self.map_array[i] < self.map_array[i+1] 
                  for i in range(len(self.map_array)-1)):
            return False
            
        if original is not None:
            # Compare with original matrix
            decompressed = self.decompress_matrix()
            return np.allclose(original, decompressed)
            
        return True
    
    def clear(self):
        """Clear all stored data."""
        self.compressed_data.clear()
        self.initialize_mapping()
        
    def get_storage_format(self) -> str:
        """Get description of storage format.
        
        Returns:
            str: Description of storage format
        """
        return f"""Matrix Compression Format:
Size: {self.size}x{self.size}
Mapping Array: Length {len(self.map_array)}
Stored Elements: {sum(len(row) for row in self.compressed_data.values())}
Format: Compressed State Mapping with Weight-Based Indexing"""

class BatchMatrixCompressor:
    """Handles compression of multiple related matrices."""
    
    def __init__(self, N: int):
        """Initialize batch compressor.
        
        Args:
            N (int): Number of units/dimensions
        """
        self.N = N
        self.compressors = {}
        
    def add_matrix(self, name: str, matrix: np.ndarray):
        """Add matrix to batch compression.
        
        Args:
            name (str): Matrix identifier
            matrix (numpy.ndarray): Matrix to compress
        """
        compressor = MatrixCompressor(self.N)
        compressor.compress_matrix(matrix)
        self.compressors[name] = compressor
        
    def get_matrix(self, name: str) -> Optional[np.ndarray]:
        """Retrieve decompressed matrix.
        
        Args:
            name (str): Matrix identifier
            
        Returns:
            Optional[numpy.ndarray]: Decompressed matrix if found
        """
        if name in self.compressors:
            return self.compressors[name].decompress_matrix()
        return None
        
    def get_batch_stats(self) -> Dict:
        """Get compression statistics for all matrices.
        
        Returns:
            Dict: Batch compression statistics
        """
        stats = {}
        for name, compressor in self.compressors.items():
            stats[name] = compressor.get_matrix_stats()
        return stats