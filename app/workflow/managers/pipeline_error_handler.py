"""
Pipeline Error Handler

Centralized error handling with error tracking and recovery strategies.
"""

import logging
import threading
from enum import Enum
from typing import Callable, Any
from dataclasses import dataclass
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    stack_trace: str | None = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class PipelineErrorHandler:
    """
    Centralized error handling for the pipeline.

    Features:
    - Error tracking and reporting
    - Automatic recovery strategies
    """

    def __init__(self):
        """Initialize error handler."""
        self.logger = logging.getLogger(__name__)

        self.error_history: list[ErrorRecord] = []
        self.error_counts: dict[str, int] = {}
        self.lock = threading.RLock()

        self.recovery_strategies: dict[str, Callable] = {}

        self.logger.info("Pipeline Error Handler initialized")

    def handle_error(self,
                    component: str,
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: dict[str, Any] | None = None) -> bool:
        """
        Handle an error with appropriate strategy.

        Args:
            component: Component where error occurred
            error: The exception
            severity: Error severity
            context: Additional context

        Returns:
            bool: True if recovered, False otherwise
        """
        error_type = type(error).__name__

        record = ErrorRecord(
            timestamp=datetime.now(),
            component=component,
            error_type=error_type,
            message=str(error),
            severity=severity,
            stack_trace=self._get_stack_trace(error)
        )

        with self.lock:
            self.error_history.append(record)
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        log_func = {
            ErrorSeverity.LOW: self.logger.debug,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical
        }.get(severity, self.logger.error)

        log_func(f"Error in {component}: {error_type} - {str(error)}")

        if error_type in self.recovery_strategies:
            try:
                record.recovery_attempted = True
                self.recovery_strategies[error_type](error, context)
                record.recovery_successful = True
                self.logger.info(f"Successfully recovered from {error_type}")
                return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")

        return False

    @staticmethod
    def _get_stack_trace(error: Exception) -> str:
        """Get stack trace from exception."""
        import traceback
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))
