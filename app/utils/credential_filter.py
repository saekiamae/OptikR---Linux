"""
Credential Logging Filter for OptikR

This module provides filtering capabilities to prevent credentials from being
logged in plaintext. It detects and masks sensitive information in log messages.

Feature: optikr-refactoring-improvements
Validates: Requirements 7.6
"""

import logging
import re
from typing import Any


class CredentialFilter:
    """
    Filter that detects and masks credentials in log messages.
    
    This filter identifies potential credentials based on:
    - Common credential field names (api_key, password, token, secret, etc.)
    - Context clues in the message
    """
    
    # Patterns for credential field names
    CREDENTIAL_FIELD_PATTERNS = [
        r'api[\s_-]?key',
        r'password',
        r'passwd',
        r'pwd',
        r'secret',
        r'token',
        r'auth',
        r'credential',
        r'access[\s_-]?key',
        r'private[\s_-]?key',
        r'client[\s_-]?secret',
    ]
    
    # Mask to use for redacted credentials
    MASK = '***REDACTED***'
    
    def __init__(self):
        """Initialize the credential filter."""
        # Compile credential field patterns
        self.field_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CREDENTIAL_FIELD_PATTERNS
        ]
    
    def filter_message(self, message: str) -> str:
        """
        Filter a log message to mask any credentials.
        
        Args:
            message: The log message to filter
            
        Returns:
            Filtered message with credentials masked
        """
        if not message:
            return message
        
        filtered = message
        
        # Check for credential field patterns in the message
        for pattern in self.field_patterns:
            # Pattern 1: key=value or key:value (captures until whitespace or end)
            filtered = re.sub(
                rf'({pattern.pattern})\s*[=:]\s*["\']?(\S+)["\']?',
                rf'\1={self.MASK}',
                filtered,
                flags=re.IGNORECASE
            )
            
            # Pattern 2: "key is value" or "key is: value"
            filtered = re.sub(
                rf'({pattern.pattern})\s+is\s*:?\s+["\']?(\S+)["\']?',
                rf'\1 is {self.MASK}',
                filtered,
                flags=re.IGNORECASE
            )
            
            # Pattern 3: "key value:" or "key value"
            filtered = re.sub(
                rf'({pattern.pattern})\s+value\s*:?\s+["\']?(\S+)["\']?',
                rf'\1 value: {self.MASK}',
                filtered,
                flags=re.IGNORECASE
            )
        
        return filtered
    
    def filter_context(self, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Filter a context dictionary to mask any credentials.
        
        Args:
            context: The context dictionary to filter
            
        Returns:
            Filtered context with credentials masked
        """
        if not context:
            return context
        
        filtered_context = {}
        
        for key, value in context.items():
            # Check if key matches credential patterns
            if self._is_credential_key(key):
                filtered_context[key] = self.MASK
            elif isinstance(value, dict):
                # Recursively filter nested dictionaries
                filtered_context[key] = self.filter_context(value)
            elif isinstance(value, list):
                # Filter list items that are strings or dicts
                filtered_context[key] = [
                    self.filter_context(item) if isinstance(item, dict)
                    else self.filter_message(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                # Filter string values that might contain credentials
                filtered_context[key] = self.filter_message(value)
            else:
                # Keep other types as-is
                filtered_context[key] = value
        
        return filtered_context
    
    def _is_credential_key(self, key: str) -> bool:
        """
        Check if a key name indicates it contains a credential.
        
        Args:
            key: The key name to check
            
        Returns:
            True if the key likely contains a credential
        """
        key_lower = key.lower()
        
        for pattern in self.field_patterns:
            if pattern.search(key_lower):
                return True
        
        return False
    
    def filter_log_entry(self, message: str, context: dict[str, Any] | None = None) -> tuple[str, dict[str, Any] | None]:
        """
        Filter both message and context for credentials.
        
        Args:
            message: The log message
            context: Optional context dictionary
            
        Returns:
            Tuple of (filtered_message, filtered_context)
        """
        filtered_message = self.filter_message(message)
        filtered_context = self.filter_context(context) if context else None
        
        return filtered_message, filtered_context


class CredentialLoggingFilter(logging.Filter):
    """logging.Filter adapter that redacts credentials from log records."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._cred_filter = CredentialFilter()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            try:
                record.msg = record.msg % record.args
                record.args = None
            except (TypeError, ValueError):
                pass
        record.msg = self._cred_filter.filter_message(str(record.msg))
        return True


# Global credential filter instance
_global_filter: CredentialFilter | None = None


def get_credential_filter() -> CredentialFilter:
    """
    Get the global credential filter instance.
    
    Returns:
        The global CredentialFilter instance
    """
    global _global_filter
    if _global_filter is None:
        _global_filter = CredentialFilter()
    return _global_filter
