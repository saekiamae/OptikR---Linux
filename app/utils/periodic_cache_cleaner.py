"""
Periodic Cache Cleaner

Automatically clears stale cache/temp files when the retention period has
elapsed.  Designed to be called at startup (synchronous) and optionally on a
timer during runtime.

Controlled by two config keys:
    storage.periodic_cache_clear_enabled  (bool, default False)
    storage.retention_days                (int,  default 30)

State key written back to config:
    storage.last_cache_clear              (ISO-8601 timestamp)

The Smart Dictionary is never affected.
"""

import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.utils.path_utils import get_cache_dir, get_temp_dir, get_logs_dir

logger = logging.getLogger(__name__)


def _parse_last_clear(raw: str | None) -> datetime | None:
    """Parse the stored ISO timestamp, returning *None* on any error."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning("Could not parse storage.last_cache_clear: %r", raw)
        return None


def _needs_clear(config_manager, *, retention_days: int) -> bool:
    """Return *True* when a periodic clear is due."""
    last_raw = config_manager.get_setting("storage.last_cache_clear", None)
    last_dt = _parse_last_clear(last_raw)
    if last_dt is None:
        return True
    now = datetime.now(timezone.utc)
    return (now - last_dt) >= timedelta(days=retention_days)


def _clear_directory_contents(directory: Path) -> tuple[int, int]:
    """Delete every item inside *directory* (but not the directory itself).

    Returns (deleted_count, failed_count).
    """
    deleted = failed = 0
    if not directory.exists() or not directory.is_dir():
        return deleted, failed
    for item in directory.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                deleted += 1
            elif item.is_dir():
                shutil.rmtree(item)
                deleted += 1
        except Exception as exc:
            logger.debug("Failed to delete %s: %s", item, exc)
            failed += 1
    return deleted, failed


def _clear_old_logs(retention_days: int) -> tuple[int, int]:
    """Delete log files older than *retention_days*."""
    deleted = failed = 0
    logs_dir = get_logs_dir()
    if not logs_dir.exists():
        return deleted, failed
    cutoff = time.time() - (retention_days * 86400)
    for log_file in logs_dir.glob("*.log*"):
        try:
            if log_file.is_file() and log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                deleted += 1
        except Exception as exc:
            logger.debug("Failed to delete old log %s: %s", log_file, exc)
            failed += 1
    return deleted, failed


def run_periodic_clear(config_manager) -> bool:
    """Check config and clear stale files if the retention period elapsed.

    Returns *True* if a clear was performed, *False* otherwise.
    """
    if not config_manager.get_setting("storage.periodic_cache_clear_enabled", False):
        return False

    retention_days = config_manager.get_setting("storage.retention_days", 30)
    if retention_days < 1:
        retention_days = 1

    if not _needs_clear(config_manager, retention_days=retention_days):
        logger.debug("Periodic cache clear: not due yet")
        return False

    logger.info("Periodic cache clear triggered (retention=%d days)", retention_days)
    total_deleted = total_failed = 0

    d, f = _clear_directory_contents(get_cache_dir())
    total_deleted += d
    total_failed += f

    d, f = _clear_directory_contents(get_temp_dir())
    total_deleted += d
    total_failed += f

    d, f = _clear_old_logs(retention_days)
    total_deleted += d
    total_failed += f

    config_manager.set_setting(
        "storage.last_cache_clear",
        datetime.now(timezone.utc).isoformat(),
    )
    config_manager.save_config()

    logger.info(
        "Periodic cache clear complete: %d items deleted, %d failed",
        total_deleted,
        total_failed,
    )
    return True
