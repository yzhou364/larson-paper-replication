import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict
from datetime import datetime
import json
from dataclasses import dataclass
import threading
from queue import Queue
import traceback

@dataclass
class LogConfig:
    """Configuration for logging setup."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    console_output: bool = True
    file_output: bool = True
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    capture_warnings: bool = True
    include_process_info: bool = False

class ModelLogger:
    """Custom logger for hypercube model."""
    
    def __init__(self, name: str = "HypercubeModel", config: Optional[LogConfig] = None):
        """Initialize model logger.
        
        Args:
            name (str): Logger name
            config (Optional[LogConfig]): Logging configuration
        """
        self.name = name
        self.config = config or LogConfig()
        self.logger = self._setup_logger()
        self._setup_exception_handling()
        self.event_queue = Queue()
        self.performance_stats = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with specified configuration.
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.config.log_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            fmt=self.config.format_string,
            datefmt=self.config.date_format
        )
        
        # Add console handler if requested
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if requested
        if self.config.file_output and self.config.log_file:
            from logging.handlers import RotatingFileHandler
            
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Configure warning capture
        if self.config.capture_warnings:
            logging.captureWarnings(True)
        
        return logger
    
    def _setup_exception_handling(self):
        """Setup global exception handling."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Handle keyboard interrupt normally
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
                
            # Log the exception
            self.logger.error(
                "Uncaught exception:",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            
        sys.excepthook = handle_exception
        
    def log_event(self, event_type: str, event_data: Dict):
        """Log model event with data.
        
        Args:
            event_type (str): Type of event
            event_data (Dict): Event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': event_data
        }
        
        if self.config.include_process_info:
            event['process'] = {
                'thread': threading.current_thread().name,
                'pid': threading.current_thread().ident
            }
            
        self.event_queue.put(event)
        self.logger.info(f"Event: {event_type}")
        
    def log_performance(self, operation: str, duration: float):
        """Log performance statistics.
        
        Args:
            operation (str): Operation name
            duration (float): Operation duration in seconds
        """
        if operation not in self.performance_stats:
            self.performance_stats[operation] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': float('-inf')
            }
            
        stats = self.performance_stats[operation]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        
        self.logger.debug(
            f"Performance - {operation}: {duration:.4f}s"
        )
        
    def dump_events(self, output_file: Union[str, Path]):
        """Dump all logged events to file.
        
        Args:
            output_file (Union[str, Path]): Output file path
        """
        events = []
        while not self.event_queue.empty():
            events.append(self.event_queue.get())
            
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2)
            
        self.logger.info(f"Events dumped to {output_file}")
        
    def get_performance_summary(self) -> Dict:
        """Get performance statistics summary.
        
        Returns:
            Dict: Performance summary
        """
        summary = {}
        for operation, stats in self.performance_stats.items():
            if stats['count'] > 0:
                summary[operation] = {
                    'count': stats['count'],
                    'total_time': stats['total_time'],
                    'average_time': stats['total_time'] / stats['count'],
                    'min_time': stats['min_time'],
                    'max_time': stats['max_time']
                }
        return summary
    
    def log_error_with_context(self, error: Exception, context: Dict):
        """Log error with additional context.
        
        Args:
            error (Exception): Error to log
            context (Dict): Error context
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        self.logger.error(
            "Error occurred with context",
            extra={'error_info': error_info}
        )
        
    def start_operation(self, operation: str) -> None:
        """Mark the start of an operation for timing.
        
        Args:
            operation (str): Operation name
        """
        setattr(threading.current_thread(), f"{operation}_start", datetime.now())
        
    def end_operation(self, operation: str) -> None:
        """Mark the end of an operation and log timing.
        
        Args:
            operation (str): Operation name
        """
        start_time = getattr(threading.current_thread(), f"{operation}_start", None)
        if start_time:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_performance(operation, duration)
            delattr(threading.current_thread(), f"{operation}_start")
            
    class OperationTimer:
        """Context manager for timing operations."""
        
        def __init__(self, logger: 'ModelLogger', operation: str):
            self.logger = logger
            self.operation = operation
            
        def __enter__(self):
            self.logger.start_operation(self.operation)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.end_operation(self.operation)
            if exc_type is not None:
                self.logger.log_error_with_context(
                    exc_val,
                    {'operation': self.operation}
                )
                
    def time_operation(self, operation: str):
        """Get context manager for timing operations.
        
        Args:
            operation (str): Operation name
            
        Returns:
            OperationTimer: Timer context manager
        """
        return self.OperationTimer(self, operation)
        
    def setup_file_logging(self, log_dir: Union[str, Path]):
        """Setup logging to file.
        
        Args:
            log_dir (Union[str, Path]): Logging directory
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.name}_{timestamp}.log"
        
        self.config.log_file = str(log_file)
        self.config.file_output = True
        
        # Reconfigure logger
        self.logger = self._setup_logger()