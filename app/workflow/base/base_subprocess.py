"""
Base Subprocess - Foundation for all pipeline stage subprocesses.

Provides:
- Subprocess lifecycle management (start/stop/restart)
- JSON message communication (stdin/stdout)
- Crash detection and automatic restart
- Error handling and logging
"""

import logging
import subprocess
import json
import threading
import queue
import time
import sys
import os
from typing import Any
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# Import EXE compatibility helpers
try:
    from ..exe_compat import get_subprocess_args
except ImportError:
    # Fallback if exe_compat not available
    def get_subprocess_args(worker_script):
        return [sys.executable, worker_script]


class BaseSubprocess(ABC):
    """Base class for all pipeline stage subprocesses."""
    
    def __init__(self, name: str, worker_script: str):
        """
        Initialize subprocess wrapper.
        
        Args:
            name: Human-readable name for this subprocess
            worker_script: Path to worker script (relative to project root)
        """
        self.name = name
        self.worker_script = worker_script
        self.process: subprocess.Popen | None = None
        self.running = False
        self.crashed = False
        self.restart_count = 0
        self.max_restarts = 3
        self.last_config = {}
        
        # Communication
        self.reader_thread: threading.Thread | None = None
        self.output_queue = queue.Queue()
        
        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.errors_count = 0
        self.start_time = None

        # Parent-death detection
        self._parent_pid: int | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_interval = 2.0  # seconds
    
    def start(self, config: dict) -> bool:
        """
        Start the subprocess.
        
        Args:
            config: Configuration dictionary for initialization
            
        Returns:
            True if started successfully
        """
        try:
            logger.info("[%s] Starting subprocess...", self.name)
            
            # Store config for potential restart
            self.last_config = config
            
            # Get subprocess arguments (EXE-compatible)
            args = get_subprocess_args(self.worker_script)
            
            # Start subprocess
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Send initial configuration
            self._send_message({'type': 'init', 'config': config})
            
            # Wait for ready signal (with timeout)
            response = self._receive_message_sync(timeout=10.0)
            
            if response and response.get('type') == 'ready':
                self.running = True
                self.crashed = False
                self.start_time = time.time()
                logger.info("[%s] Subprocess ready (PID: %s)", self.name, self.process.pid)
                
                # Start background reader thread
                self.reader_thread = threading.Thread(
                    target=self._read_output_loop,
                    daemon=True,
                    name=f"{self.name}-Reader"
                )
                self.reader_thread.start()
                
                # Start parent-death watchdog
                self._parent_pid = os.getpid()
                self._watchdog_thread = threading.Thread(
                    target=self._parent_death_watchdog,
                    daemon=True,
                    name=f"{self.name}-Watchdog"
                )
                self._watchdog_thread.start()
                
                return True
            else:
                logger.error("[%s] Subprocess failed to initialize", self.name)
                logger.error("[%s] Response: %s", self.name, response)
                
                # Try to read stderr for error messages
                if self.process and self.process.stderr:
                    try:
                        stderr_output = self.process.stderr.read()
                        if stderr_output:
                            logger.error("[%s] STDERR: %s", self.name, stderr_output)
                    except Exception:
                        pass
                
                self._cleanup_process()
                return False
                
        except Exception as e:
            logger.error("[%s] Failed to start: %s", self.name, e, exc_info=True)
            self._cleanup_process()
            return False
    
    def stop(self):
        """Stop the subprocess gracefully."""
        if not self.process:
            return
        
        logger.info("[%s] Stopping subprocess...", self.name)
        self.running = False
        
        try:
            # Send shutdown message
            self._send_message({'type': 'shutdown'})
            
            # Wait for process to exit
            self.process.wait(timeout=5.0)
            logger.info("[%s] Subprocess stopped gracefully", self.name)
            
        except subprocess.TimeoutExpired:
            logger.warning("[%s] Subprocess didn't stop, terminating...", self.name)
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] Force killing subprocess...", self.name)
                self.process.kill()
        
        except Exception as e:
            logger.error("[%s] Error during stop: %s", self.name, e)
        
        finally:
            self._cleanup_process()
    
    def process_data(self, data: Any, timeout: float = 5.0) -> Any | None:
        """
        Send data to subprocess and get result.
        
        Args:
            data: Data to process
            timeout: Maximum time to wait for result (seconds)
            
        Returns:
            Processed result or None on error
        """
        if not self.running:
            if self.crashed and self.restart_count < self.max_restarts:
                logger.info("[%s] Attempting restart (%d/%d)...", self.name, self.restart_count + 1, self.max_restarts)
                if self.restart():
                    self.restart_count += 1
                else:
                    logger.error("[%s] Restart failed", self.name)
                    return None
            else:
                logger.warning("[%s] Subprocess not running", self.name)
                return None
        
        try:
            # Prepare message data
            prepared_data = self._prepare_message(data)
            
            # Create process message with data field
            message = {
                'type': 'process',
                'data': prepared_data
            }
            
            # Send data
            self._send_message(message)
            
            # Wait for result using deadline-based timeout
            deadline = time.monotonic() + timeout
            remaining = timeout
            while remaining > 0:
                try:
                    result = self.output_queue.get(timeout=remaining)
                    
                    if result.get('type') == 'result':
                        self.messages_received += 1
                        return self._parse_result(result)
                    elif result.get('type') == 'error':
                        self.errors_count += 1
                        logger.error("[%s] Error: %s", self.name, result.get('error'))
                        return None
                    else:
                        self.output_queue.put(result)
                        
                except queue.Empty:
                    pass
                remaining = deadline - time.monotonic()
            
            logger.warning("[%s] Timeout waiting for result", self.name)
            return None
                
        except Exception as e:
            logger.error("[%s] Processing error: %s", self.name, e, exc_info=True)
            self.crashed = True
            return None
    
    def restart(self) -> bool:
        """Restart crashed subprocess."""
        logger.info("[%s] Restarting subprocess...", self.name)
        self.stop()
        time.sleep(0.5)  # Brief pause before restart
        return self.start(self.last_config)
    
    def is_alive(self) -> bool:
        """Check if subprocess is alive."""
        if not self.process:
            return False
        return self.process.poll() is None
    
    def _send_message(self, message: dict):
        """Send JSON message to subprocess."""
        try:
            if not self.process or not self.process.stdin:
                raise RuntimeError("Process not running")
            
            json_str = json.dumps(message)
            self.process.stdin.write(json_str + '\n')
            self.process.stdin.flush()
            self.messages_sent += 1
            
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            self.crashed = True
            raise
    
    def _receive_message_sync(self, timeout: float = 5.0) -> dict | None:
        """Receive JSON message from subprocess (synchronous with timeout).

        Uses a background thread to avoid blocking indefinitely on readline().
        """
        try:
            if not self.process or not self.process.stdout:
                return None

            result_queue: queue.Queue = queue.Queue()

            def _reader():
                try:
                    line = self.process.stdout.readline()
                    if line:
                        result_queue.put(json.loads(line.strip()))
                    else:
                        result_queue.put(None)
                except Exception as exc:
                    result_queue.put(exc)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()

            try:
                result = result_queue.get(timeout=timeout)
            except queue.Empty:
                return None

            if isinstance(result, Exception):
                logger.error("[%s] Receive error: %s", self.name, result)
                return None
            return result

        except Exception as e:
            logger.error("[%s] Receive error: %s", self.name, e)
            return None
    
    def _read_output_loop(self):
        """Background thread to read subprocess output."""
        while self.running and self.is_alive():
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                # Parse JSON message
                message = json.loads(line.strip())
                self.output_queue.put(message)
                
            except json.JSONDecodeError as e:
                logger.warning("[%s] Invalid JSON: %s", self.name, line)
            except Exception as e:
                logger.error("[%s] Reader error: %s", self.name, e)
                break
        
        logger.debug("[%s] Reader thread stopped", self.name)

    def _parent_death_watchdog(self):
        """Background thread that monitors parent process liveness."""
        while self.running and self.is_alive():
            try:
                # Check if parent PID is still alive
                os.kill(self._parent_pid, 0)
            except OSError:
                # Parent is dead — self-terminate
                logger.warning("[%s] Parent process (PID %s) is dead, self-terminating", self.name, self._parent_pid)
                self.stop()
                return
            time.sleep(self._watchdog_interval)

    
    def _cleanup_process(self):
        """Clean up process resources.

        Closes stdout first to unblock the reader thread (which may be
        stuck on ``readline()`` — especially common on Windows), then
        joins background threads, and finally closes remaining streams.
        """
        self.running = False

        # Close stdout before joining the reader thread so readline()
        # returns immediately instead of blocking until the join timeout.
        if self.process:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2.0)
            self.reader_thread = None

        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception:
                pass
            
            self.process = None
    
    @abstractmethod
    def _prepare_message(self, data: Any) -> dict:
        """
        Prepare data for sending to subprocess.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary to send as JSON
        """
        raise NotImplementedError("Subclasses must implement _prepare_message()")
    
    @abstractmethod
    def _parse_result(self, result: dict) -> Any:
        """
        Parse result from subprocess.
        
        Args:
            result: Result dictionary from subprocess
            
        Returns:
            Parsed result data
        """
        raise NotImplementedError("Subclasses must implement _parse_result()")
    
    def get_metrics(self) -> dict:
        """Get subprocess metrics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'name': self.name,
            'running': self.running,
            'crashed': self.crashed,
            'restart_count': self.restart_count,
            'pid': self.process.pid if self.process else None,
            'uptime_seconds': uptime,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'errors_count': self.errors_count,
        }
