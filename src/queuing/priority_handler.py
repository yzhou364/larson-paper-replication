import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PriorityCall:
    """Represents a prioritized call in the system."""
    id: str
    atom: int
    priority: int
    entry_time: float
    preempted: bool = False
    preemption_count: int = 0
    total_wait_time: float = 0.0
    service_start_time: Optional[float] = None
    assigned_unit: Optional[int] = None

class PriorityHandler:
    """Handles priority-based call management in hypercube model."""
    
    def __init__(self, N: int, num_priorities: int = 3, 
                 allow_preemption: bool = True):
        """Initialize priority handler.
        
        Args:
            N (int): Number of units
            num_priorities (int): Number of priority levels
            allow_preemption (bool): Whether to allow preemption
        """
        self.N = N
        self.num_priorities = num_priorities
        self.allow_preemption = allow_preemption
        
        # Priority queues (higher index = higher priority)
        self.priority_queues = [deque() for _ in range(num_priorities)]
        
        # Active calls being serviced
        self.active_calls = {}  # call_id -> PriorityCall
        
        # Statistics
        self.stats = self._initialize_stats()
        
        # Call history
        self.call_history = []
        
    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            'by_priority': [{
                'total_calls': 0,
                'queued_calls': 0,
                'preempted_calls': 0,
                'total_wait': 0.0,
                'max_wait': 0.0,
                'total_service': 0.0,
                'completed_calls': 0
            } for _ in range(self.num_priorities)],
            'total_preemptions': 0,
            'max_queue_length': 0,
            'total_calls': 0
        }
    
    def add_call(self, atom: int, priority: int, time: float) -> PriorityCall:
        """Add a new call to the system.
        
        Args:
            atom (int): Calling atom
            priority (int): Call priority level
            time (float): Current time
            
        Returns:
            PriorityCall: Created call object
        """
        if priority >= self.num_priorities:
            raise ValueError(f"Invalid priority level: {priority}")
            
        call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        call = PriorityCall(
            id=call_id,
            atom=atom,
            priority=priority,
            entry_time=time
        )
        
        # Update statistics
        self.stats['total_calls'] += 1
        self.stats['by_priority'][priority]['total_calls'] += 1
        
        # Add to appropriate queue
        self.priority_queues[priority].append(call)
        self.stats['by_priority'][priority]['queued_calls'] += 1
        
        # Update max queue length
        total_queued = sum(len(q) for q in self.priority_queues)
        self.stats['max_queue_length'] = max(
            self.stats['max_queue_length'],
            total_queued
        )
        
        return call
    
    def get_next_call(self, time: float) -> Optional[PriorityCall]:
        """Get highest priority waiting call.
        
        Args:
            time (float): Current time
            
        Returns:
            Optional[PriorityCall]: Next call to service
        """
        # Check queues from highest to lowest priority
        for priority in range(self.num_priorities - 1, -1, -1):
            if self.priority_queues[priority]:
                call = self.priority_queues[priority].popleft()
                wait_time = time - call.entry_time
                
                # Update statistics
                call.total_wait_time += wait_time
                call.service_start_time = time
                self.stats['by_priority'][priority]['total_wait'] += wait_time
                self.stats['by_priority'][priority]['max_wait'] = max(
                    self.stats['by_priority'][priority]['max_wait'],
                    wait_time
                )
                
                return call
                
        return None
    
    def complete_call(self, call: PriorityCall, time: float):
        """Mark a call as completed.
        
        Args:
            call (PriorityCall): Call to complete
            time (float): Current time
        """
        if call.id in self.active_calls:
            del self.active_calls[call.id]
        
        # Update statistics
        priority = call.priority
        self.stats['by_priority'][priority]['completed_calls'] += 1
        self.stats['by_priority'][priority]['total_service'] += (
            time - call.service_start_time
        )
        
        # Add to history
        self.call_history.append({
            'call_id': call.id,
            'priority': call.priority,
            'atom': call.atom,
            'entry_time': call.entry_time,
            'completion_time': time,
            'wait_time': call.total_wait_time,
            'service_time': time - call.service_start_time,
            'preempted': call.preempted,
            'preemption_count': call.preemption_count
        })
    
    def attempt_preemption(self, new_call: PriorityCall, 
                          current_time: float) -> Optional[Tuple[PriorityCall, int]]:
        """Attempt to preempt a lower priority call.
        
        Args:
            new_call (PriorityCall): New high-priority call
            current_time (float): Current time
            
        Returns:
            Optional[Tuple[PriorityCall, int]]: Preempted call and its unit if successful
        """
        if not self.allow_preemption:
            return None
            
        # Find lowest priority active call
        lowest_priority = float('inf')
        target_call = None
        target_unit = None
        
        for call_id, active_call in self.active_calls.items():
            if (active_call.priority < new_call.priority and 
                active_call.priority < lowest_priority):
                lowest_priority = active_call.priority
                target_call = active_call
                target_unit = active_call.assigned_unit
        
        if target_call is not None:
            # Preempt the call
            target_call.preempted = True
            target_call.preemption_count += 1
            target_call.total_wait_time += (
                current_time - target_call.service_start_time
            )
            
            # Update statistics
            self.stats['total_preemptions'] += 1
            self.stats['by_priority'][target_call.priority]['preempted_calls'] += 1
            
            # Return call to appropriate queue
            self.priority_queues[target_call.priority].append(target_call)
            
            return target_call, target_unit
            
        return None
    
    def get_queue_lengths(self) -> List[int]:
        """Get current queue length for each priority level.
        
        Returns:
            List[int]: Queue lengths by priority
        """
        return [len(queue) for queue in self.priority_queues]
    
    def compute_priority_metrics(self) -> Dict:
        """Compute performance metrics by priority level.
        
        Returns:
            Dict: Priority performance metrics
        """
        metrics = {
            'by_priority': [],
            'overall': {
                'total_calls': self.stats['total_calls'],
                'total_preemptions': self.stats['total_preemptions'],
                'max_queue_length': self.stats['max_queue_length']
            }
        }
        
        for priority in range(self.num_priorities):
            pstats = self.stats['by_priority'][priority]
            
            if pstats['total_calls'] > 0:
                metrics['by_priority'].append({
                    'priority': priority,
                    'total_calls': pstats['total_calls'],
                    'completion_rate': (
                        pstats['completed_calls'] / pstats['total_calls']
                    ),
                    'average_wait': (
                        pstats['total_wait'] / pstats['total_calls']
                    ),
                    'average_service': (
                        pstats['total_service'] / pstats['completed_calls']
                        if pstats['completed_calls'] > 0 else 0
                    ),
                    'preemption_rate': (
                        pstats['preempted_calls'] / pstats['total_calls']
                    ),
                    'max_wait': pstats['max_wait']
                })
        
        return metrics
    
    def get_historical_analysis(self) -> Dict:
        """Analyze historical call data.
        
        Returns:
            Dict: Historical performance analysis
        """
        if not self.call_history:
            return {}
            
        analysis = {
            'by_priority': defaultdict(list),
            'overall': {
                'total_calls': len(self.call_history),
                'average_wait': np.mean([
                    call['wait_time'] for call in self.call_history
                ]),
                'average_service': np.mean([
                    call['service_time'] for call in self.call_history
                ]),
                'preemption_rate': np.mean([
                    call['preempted'] for call in self.call_history
                ])
            }
        }
        
        # Analyze by priority
        for call in self.call_history:
            priority = call['priority']
            analysis['by_priority'][priority].append({
                'wait_time': call['wait_time'],
                'service_time': call['service_time'],
                'preempted': call['preempted'],
                'preemption_count': call['preemption_count']
            })
        
        # Compute priority-specific metrics
        for priority, calls in analysis['by_priority'].items():
            analysis['by_priority'][priority] = {
                'count': len(calls),
                'average_wait': np.mean([c['wait_time'] for c in calls]),
                'average_service': np.mean([c['service_time'] for c in calls]),
                'preemption_rate': np.mean([c['preempted'] for c in calls]),
                'average_preemptions': np.mean([c['preemption_count'] for c in calls])
            }
        
        return analysis
    
    def clear_queues(self):
        """Clear all queues and reset statistics."""
        # Clear queues
        for queue in self.priority_queues:
            queue.clear()
        
        # Clear active calls
        self.active_calls.clear()
        
        # Reset statistics
        self.stats = self._initialize_stats()
        
        # Clear history
        self.call_history.clear()