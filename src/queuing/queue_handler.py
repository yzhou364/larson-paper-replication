import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque

class QueueHandler:
    """Handles queue operations for hypercube model."""
    
    def __init__(self, N: int, capacity: str = 'infinite'):
        """
        Initialize queue handler.
        
        Args:
            N (int): Number of units
            capacity (str): Queue capacity type ('zero' or 'infinite')
        """
        self.N = N
        self.capacity = capacity
        self.queue = deque() if capacity == 'infinite' else None
        self.queue_stats = {
            'total_queued': 0,
            'max_queue_length': 0,
            'total_queue_time': 0
        }
        
    def can_queue(self, state: List[int]) -> bool:
        """Check if system can queue new calls in current state."""
        if self.capacity == 'zero':
            return False
        return sum(state) == self.N  # All units busy
        
    def add_to_queue(self, call: Dict, time: float) -> bool:
        """Add call to queue if possible."""
        if not self.can_queue(call['state']):
            return False
            
        if self.capacity == 'infinite':
            call['queue_entry_time'] = time
            self.queue.append(call)
            self.queue_stats['total_queued'] += 1
            self.queue_stats['max_queue_length'] = max(
                self.queue_stats['max_queue_length'],
                len(self.queue)
            )
            return True
        return False
        
    def get_next_call(self) -> Optional[Dict]:
        """Get next call from queue (FCFS)."""
        if self.queue and self.capacity == 'infinite':
            return self.queue.popleft()
        return None
        
    def update_queue_times(self, current_time: float):
        """Update queue statistics."""
        if self.capacity == 'infinite' and self.queue:
            for call in self.queue:
                wait_time = current_time - call['queue_entry_time']
                self.queue_stats['total_queue_time'] += wait_time
                
    def compute_queue_metrics(self, total_time: float) -> Dict:
        """Compute queue performance metrics."""
        metrics = {
            'average_queue_length': 0,
            'average_wait_time': 0,
            'fraction_queued': 0
        }
        
        if self.capacity == 'infinite' and self.queue_stats['total_queued'] > 0:
            metrics['average_queue_length'] = (
                self.queue_stats['total_queue_time'] / total_time
            )
            metrics['average_wait_time'] = (
                self.queue_stats['total_queue_time'] / 
                self.queue_stats['total_queued']
            )
            metrics['fraction_queued'] = (
                self.queue_stats['total_queued'] / 
                (self.queue_stats['total_queued'] + total_time)
            )
            
        return metrics
    
    def get_queue_length(self) -> int:
        """Get current queue length."""
        return len(self.queue) if self.capacity == 'infinite' else 0
    
    def clear_queue(self):
        """Clear all calls from queue."""
        if self.capacity == 'infinite':
            self.queue.clear()
            
    def reset_stats(self):
        """Reset queue statistics."""
        self.queue_stats = {
            'total_queued': 0,
            'max_queue_length': 0,
            'total_queue_time': 0
        }