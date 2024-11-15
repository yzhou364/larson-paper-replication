import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class QueuedCall:
    """Represents a call in the queue."""
    atom: int  # Originating atom
    entry_time: float  # Time call entered queue
    priority: int = 0  # Priority level (0 = lowest)
    id: str = None  # Unique call identifier
    
    def __post_init__(self):
        """Initialize call ID if not provided."""
        if self.id is None:
            self.id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

class QueueHandler:
    """Handles queue operations for hypercube model."""
    
    def __init__(self, N: int, capacity: str = 'infinite', num_priorities: int = 1):
        """Initialize queue handler.
        
        Args:
            N (int): Number of units
            capacity (str): Queue capacity type ('zero' or 'infinite')
            num_priorities (int): Number of priority levels
        """
        self.N = N
        self.capacity = capacity
        self.num_priorities = num_priorities
        
        # Initialize queues for each priority level
        self.queues = [deque() for _ in range(num_priorities)]
        
        # Statistics tracking
        self.stats = {
            'total_arrivals': 0,
            'total_queued': 0,
            'max_queue_length': 0,
            'total_queue_time': 0,
            'by_priority': [{
                'total_queued': 0,
                'total_wait': 0,
                'max_wait': 0
            } for _ in range(num_priorities)]
        }
        
        # Historical data
        self.history = []
    
    def can_queue(self, state: List[int]) -> bool:
        """Check if system can queue new calls in current state.
        
        Args:
            state (List[int]): Current system state
            
        Returns:
            bool: True if system can queue calls
        """
        if self.capacity == 'zero':
            return False
        return sum(state) == self.N  # All units busy
    
    def add_to_queue(self, atom: int, time: float, priority: int = 0) -> Optional[QueuedCall]:
        """Add call to queue.
        
        Args:
            atom (int): Calling atom
            time (float): Current time
            priority (int): Call priority level
            
        Returns:
            Optional[QueuedCall]: Queued call if successful, None otherwise
        """
        if self.capacity == 'zero':
            return None
            
        if priority >= self.num_priorities:
            raise ValueError(f"Invalid priority level: {priority}")
        
        # Create queued call
        call = QueuedCall(atom=atom, entry_time=time, priority=priority)
        
        # Add to appropriate queue
        self.queues[priority].append(call)
        
        # Update statistics
        self.stats['total_arrivals'] += 1
        self.stats['total_queued'] += 1
        self.stats['max_queue_length'] = max(
            self.stats['max_queue_length'],
            self.get_total_queue_length()
        )
        self.stats['by_priority'][priority]['total_queued'] += 1
        
        return call
    
    def get_next_call(self, current_time: float) -> Optional[QueuedCall]:
        """Get next call from queue (highest priority first).
        
        Args:
            current_time (float): Current time
            
        Returns:
            Optional[QueuedCall]: Next call to service, None if queue empty
        """
        # Check queues from highest to lowest priority
        for priority in range(self.num_priorities - 1, -1, -1):
            if self.queues[priority]:
                call = self.queues[priority].popleft()
                wait_time = current_time - call.entry_time
                
                # Update statistics
                self.stats['total_queue_time'] += wait_time
                self.stats['by_priority'][priority]['total_wait'] += wait_time
                self.stats['by_priority'][priority]['max_wait'] = max(
                    self.stats['by_priority'][priority]['max_wait'],
                    wait_time
                )
                
                # Record in history
                self.history.append({
                    'call_id': call.id,
                    'atom': call.atom,
                    'priority': call.priority,
                    'entry_time': call.entry_time,
                    'exit_time': current_time,
                    'wait_time': wait_time
                })
                
                return call
                
        return None
    
    def get_queue_lengths(self) -> List[int]:
        """Get current queue length for each priority level.
        
        Returns:
            List[int]: Queue lengths by priority
        """
        return [len(queue) for queue in self.queues]
    
    def get_total_queue_length(self) -> int:
        """Get total number of calls in queue.
        
        Returns:
            int: Total queue length
        """
        return sum(len(queue) for queue in self.queues)
    
    def compute_queue_metrics(self) -> Dict:
        """Compute queue performance metrics.
        
        Returns:
            Dict: Queue performance metrics
        """
        metrics = {
            'total_queued': self.stats['total_queued'],
            'max_queue_length': self.stats['max_queue_length'],
            'average_wait': (
                self.stats['total_queue_time'] / self.stats['total_queued']
                if self.stats['total_queued'] > 0 else 0
            ),
            'queue_probability': (
                self.stats['total_queued'] / self.stats['total_arrivals']
                if self.stats['total_arrivals'] > 0 else 0
            ),
            'by_priority': []
        }
        
        # Compute metrics for each priority level
        for priority in range(self.num_priorities):
            priority_stats = self.stats['by_priority'][priority]
            metrics['by_priority'].append({
                'total_queued': priority_stats['total_queued'],
                'average_wait': (
                    priority_stats['total_wait'] / priority_stats['total_queued']
                    if priority_stats['total_queued'] > 0 else 0
                ),
                'max_wait': priority_stats['max_wait']
            })
            
        return metrics
    
    def get_historical_metrics(self) -> Dict:
        """Compute metrics from historical data.
        
        Returns:
            Dict: Historical performance metrics
        """
        if not self.history:
            return {}
            
        wait_times = [call['wait_time'] for call in self.history]
        
        return {
            'average_wait': np.mean(wait_times),
            'median_wait': np.median(wait_times),
            'max_wait': np.max(wait_times),
            'std_wait': np.std(wait_times),
            'total_calls': len(self.history),
            'by_priority': {
                priority: {
                    'count': len([
                        call for call in self.history 
                        if call['priority'] == priority
                    ]),
                    'average_wait': np.mean([
                        call['wait_time'] for call in self.history 
                        if call['priority'] == priority
                    ])
                }
                for priority in range(self.num_priorities)
            }
        }
    
    def clear_queues(self):
        """Clear all queues and reset statistics."""
        # Clear queues
        for queue in self.queues:
            queue.clear()
            
        # Reset statistics
        self.stats = {
            'total_arrivals': 0,
            'total_queued': 0,
            'max_queue_length': 0,
            'total_queue_time': 0,
            'by_priority': [{
                'total_queued': 0,
                'total_wait': 0,
                'max_wait': 0
            } for _ in range(self.num_priorities)]
        }
        
        # Clear history
        self.history.clear()