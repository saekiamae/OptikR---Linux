"""Credential encryption using Windows DPAPI.

This module provides secure credential storage using Windows Data Protection API (DPAPI).
Credentials are encrypted per-user and can only be decrypted by the same Windows user account.
"""

import base64
import logging

try:
    import win32crypt
    DPAPI_AVAILABLE = True
except ImportError:
    DPAPI_AVAILABLE = False
    logging.warning("win32crypt not available - credential encryption disabled")


logger = logging.getLogger(__name__)


class CredentialEncryptor:
    """Encrypts credentials using Windows DPAPI.
    
    This class provides methods to encrypt and decrypt sensitive credentials
    using Windows Data Protection API (DPAPI). Encrypted credentials can only
    be decrypted by the same Windows user account that encrypted them.
    
    Attributes:
        None
        
    Example:
        >>> encryptor = CredentialEncryptor()
        >>> encrypted = encryptor.encrypt("my_api_key_12345")
        >>> decrypted = encryptor.decrypt(encrypted)
        >>> assert decrypted == "my_api_key_12345"
    """
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext using DPAPI.
        
        Args:
            plaintext: The plaintext string to encrypt
            
        Returns:
            Base64-encoded encrypted data
            
        Raises:
            RuntimeError: If DPAPI is not available on this system
            ValueError: If plaintext is empty or None
            
        Example:
            >>> encryptor = CredentialEncryptor()
            >>> encrypted = encryptor.encrypt("secret_key")
            >>> print(len(encrypted) > 20)  # Encrypted data is longer
            True
        """
        if not DPAPI_AVAILABLE:
            raise RuntimeError("DPAPI not available - cannot encrypt credentials")
        
        if not plaintext:
            raise ValueError("Cannot encrypt empty or None plaintext")
        
        try:
            # Convert string to bytes
            plaintext_bytes = plaintext.encode('utf-8')
            
            # Encrypt using DPAPI
            encrypted_bytes = win32crypt.CryptProtectData(
                plaintext_bytes,
                None,  # Optional description
                None,  # Optional entropy
                None,  # Reserved
                None,  # Prompt struct
                0      # Flags
            )
            
            # Encode as base64 for storage
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode('ascii')
            
            logger.debug(f"Successfully encrypted credential (length: {len(plaintext)})")
            return encrypted_b64
            
        except Exception as e:
            logger.error(f"Failed to encrypt credential: {e}")
            raise RuntimeError(f"Encryption failed: {e}") from e
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext using DPAPI.
        
        Args:
            ciphertext: Base64-encoded encrypted data
            
        Returns:
            Decrypted plaintext string
            
        Raises:
            RuntimeError: If DPAPI is not available on this system
            ValueError: If ciphertext is empty, None, or invalid base64
            
        Example:
            >>> encryptor = CredentialEncryptor()
            >>> encrypted = encryptor.encrypt("secret_key")
            >>> decrypted = encryptor.decrypt(encrypted)
            >>> print(decrypted)
            secret_key
        """
        if not DPAPI_AVAILABLE:
            raise RuntimeError("DPAPI not available - cannot decrypt credentials")
        
        if not ciphertext:
            raise ValueError("Cannot decrypt empty or None ciphertext")
        
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(ciphertext)
            
            # Decrypt using DPAPI
            decrypted_bytes = win32crypt.CryptUnprotectData(
                encrypted_bytes,
                None,  # Optional entropy
                None,  # Reserved
                None,  # Prompt struct
                0      # Flags
            )
            
            # CryptUnprotectData returns tuple: (description, decrypted_data)
            plaintext = decrypted_bytes[1].decode('utf-8')
            
            logger.debug(f"Successfully decrypted credential (length: {len(plaintext)})")
            return plaintext
            
        except base64.binascii.Error as e:
            logger.error(f"Invalid base64 in ciphertext: {e}")
            raise ValueError(f"Invalid encrypted data format: {e}") from e
        except Exception as e:
            logger.error(f"Failed to decrypt credential: {e}")
            raise RuntimeError(f"Decryption failed: {e}") from e
    
    def is_encrypted(self, value: str) -> bool:
        """Check if value appears to be encrypted.
        
        This method performs a heuristic check to determine if a string
        looks like encrypted data. It checks for:
        - Valid base64 encoding
        - Minimum length (encrypted data is typically longer)
        - Alphanumeric characters with base64 padding
        
        Args:
            value: The string to check
            
        Returns:
            True if value appears to be encrypted, False otherwise
            
        Note:
            This is a heuristic check and may have false positives/negatives.
            It's primarily used to avoid double-encryption.
            
        Example:
            >>> encryptor = CredentialEncryptor()
            >>> print(encryptor.is_encrypted("plain_text"))
            False
            >>> encrypted = encryptor.encrypt("secret")
            >>> print(encryptor.is_encrypted(encrypted))
            True
        """
        if not value or len(value) < 20:
            return False
        
        try:
            # Try to decode as base64
            decoded = base64.b64decode(value, validate=True)
            
            # Encrypted data should be at least 20 bytes
            if len(decoded) < 20:
                return False
            
            # Check if it looks like base64 (alphanumeric + padding)
            # Base64 uses A-Z, a-z, 0-9, +, /, and = for padding
            valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
            if not all(c in valid_chars for c in value):
                return False
            
            return True
            
        except Exception:
            return False
