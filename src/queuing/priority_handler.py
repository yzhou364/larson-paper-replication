import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict, deque

class PriorityHandler:
    """Handles call priorities in hypercube model."""
    
    def __init__(self, N: int, num_priorities: int = 3):
        """
        Initialize priority handler.
        
        Args:
            N (int): Number of units
            num_priorities (int): Number of priority levels (default=3)
        """
        self.N = N
        self.num_priorities = num_priorities
        self.priority_queues = [deque() for _ in range(num_priorities)]
        self.stats = defaultdict(lambda: {
            'total_calls': 0,
            'total_wait': 0,
            'max_wait': 0
        })
        
    def add_call(self, call: Dict, time: float) -> bool:
        """Add call to appropriate priority queue."""
        priority = call['priority']
        if priority >= self.num_priorities:
            raise ValueError(f"Invalid priority level: {priority}")
            
        call['queue_entry_time'] = time
        self.priority_queues[priority].append(call)
        self.stats[priority]['total_calls'] += 1
        return True
        
    def get_next_call(self, current_time: float) -> Optional[Dict]:
        """Get highest priority waiting call."""
        # Check queues from highest to lowest priority
        for priority in range(self.num_priorities):
            if self.priority_queues[priority]:
                call = self.priority_queues[priority].popleft()
                wait_time = current_time - call['queue_entry_time']
                self.stats[priority]['total_wait'] += wait_time
                self.stats[priority]['max_wait'] = max(
                    self.stats[priority]['max_wait'],
                    wait_time
                )
                return call
        return None
        
    def get_queue_lengths(self) -> List[int]:
        """Get current queue length for each priority level."""
        return [len(queue) for queue in self.priority_queues]
        
    def compute_priority_metrics(self) -> Dict:
        """Compute performance metrics by priority level."""
        metrics = {}
        
        for priority in range(self.num_priorities):
            if self.stats[priority]['total_calls'] > 0:
                metrics[f'priority_{priority}'] = {
                    'average_wait': (
                        self.stats[priority]['total_wait'] /
                        self.stats[priority]['total_calls']
                    ),
                    'max_wait': self.stats[priority]['max_wait'],
                    'total_calls': self.stats[priority]['total_calls']
                }
            else:
                metrics[f'priority_{priority}'] = {
                    'average_wait': 0,
                    'max_wait': 0,
                    'total_calls': 0
                }
                
        return metrics
    
    def preempt_lower_priority(self, min_priority: int) -> Optional[Dict]:
        """Attempt to preempt a lower priority call."""
        for priority in range(self.num_priorities - 1, min_priority - 1, -1):
            if self.priority_queues[priority]:
                return self.priority_queues[priority].popleft()
        return None
    
    def clear_queues(self):
        """Clear all priority queues."""
        for queue in self.priority_queues:
            queue.clear()
            
    def reset_stats(self):
        """Reset priority statistics."""
        self.stats = defaultdict(lambda: {
            'total_calls': 0,
            'total_wait': 0,
            'max_wait': 0
        })