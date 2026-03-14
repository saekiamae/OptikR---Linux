"""
Pipeline execution strategies.

Each strategy controls *how* a list of stages is executed:

- ``SequentialStrategy``:  in-process, one stage after another.
- ``AsyncStrategy``:       overlapping pipeline execution with per-stage
                           worker threads (replaces AsyncPipelineOptimizer).
- ``CustomStrategy``:      per-stage choice of sequential or async execution.
- ``SubprocessStrategy``:  crash-isolated execution via ``SubprocessManager``.

Requirements: 2.2
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import Any

from app.workflow.managers.pipeline_error_handler import (
    PipelineErrorHandler,
    ErrorSeverity,
)
from .types import ExecutionMode, PipelineStageProtocol, StageResult


logger = logging.getLogger('optikr.pipeline.strategies')

_POISON = object()


# ---------------------------------------------------------------------------
# SequentialStrategy
# ---------------------------------------------------------------------------

class SequentialStrategy:
    """Execute stages sequentially in the current process.

    This is the simplest strategy -- each stage runs synchronously and
    passes its output dict to the next stage.  Stage failures are
    recorded via the optional ``PipelineErrorHandler``.
    """

    _MAX_TIMING_SAMPLES = 100

    def __init__(
        self,
        error_handler: PipelineErrorHandler | None = None,
    ) -> None:
        self._error_handler = error_handler
        self._stats_lock = threading.Lock()
        self._stage_times: dict[str, list[float]] = {}

    def run_pipeline(
        self,
        stages: list[PipelineStageProtocol],
        initial_input: dict[str, Any],
    ) -> StageResult:
        current_data = dict(initial_input)
        last_result = StageResult(success=True, data=current_data)

        for stage in stages:
            stage_name = getattr(stage, "name", type(stage).__name__)
            try:
                result = stage.execute(current_data)
            except Exception as exc:
                logger.error("[Pipeline] Stage '%s' raised: %s", stage_name, exc)
                self._record_error(stage_name, exc, ErrorSeverity.HIGH)
                return StageResult(
                    success=False,
                    data=current_data,
                    duration_ms=last_result.duration_ms,
                    error=f"{stage_name}: {exc}",
                )

            if not result.success:
                logger.warning(
                    "[Pipeline] Stage '%s' failed: %s",
                    stage_name, result.error,
                )
                self._record_error(
                    stage_name,
                    RuntimeError(result.error or "stage returned failure"),
                    ErrorSeverity.MEDIUM,
                )
                return result

            with self._stats_lock:
                times = self._stage_times.setdefault(stage_name, [])
                times.append(result.duration_ms)
                if len(times) > self._MAX_TIMING_SAMPLES:
                    del times[:len(times) - self._MAX_TIMING_SAMPLES]

            data_keys = sorted(result.data.keys()) if result.data else []
            logger.debug(
                "[Pipeline] Stage '%s' OK  %.1fms  keys=%s",
                stage_name, result.duration_ms, data_keys,
            )

            current_data.update(result.data)
            last_result = StageResult(
                success=True,
                data=current_data,
                duration_ms=last_result.duration_ms + result.duration_ms,
            )

        logger.debug(
            "[Pipeline] Frame complete  total=%.1fms  stages=%d",
            last_result.duration_ms, len(stages),
        )
        return last_result

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of sequential strategy statistics."""
        with self._stats_lock:
            avg_times: dict[str, float] = {}
            for stage_name, times in self._stage_times.items():
                if times:
                    avg_times[stage_name] = sum(times) / len(times)
            return {
                'avg_stage_times_ms': avg_times,
            }

    def _record_error(
        self,
        component: str,
        error: Exception,
        severity: ErrorSeverity,
    ) -> None:
        if self._error_handler:
            self._error_handler.handle_error(
                component=component,
                error=error,
                severity=severity,
            )
        else:
            logger.error("Stage %s failed: %s", component, error)


InProcessStrategy = SequentialStrategy


# ---------------------------------------------------------------------------
# AsyncStrategy
# ---------------------------------------------------------------------------

class AsyncStrategy:
    """Overlapping pipeline execution with per-stage worker threads.

    Each stage gets a dedicated worker thread and a bounded input queue.
    Data flows between stages via queue hand-offs, allowing stage *N+1*
    to process frame *K* while stage *N* already processes frame *K+1*.

    Pipeline-cleanup-fixes baked in from scratch:

    * **PCF 1.1** -- drain-and-retry poison-pill delivery so workers
      always receive the stop signal even when their queue is full.
    * **PCF 1.2** -- output queues are resolved *at runtime* by stage
      name, not at registration time, so registration order is
      irrelevant.
    * **PCF 1.3** -- all mutations of shared statistics are guarded by
      ``_stats_lock``.
    * **PCF 1.4** -- the internal ``ThreadPoolExecutor`` is shut down
      with ``wait=False, cancel_futures=True`` to prevent indefinite
      blocking on hung futures.
    """

    def __init__(
        self,
        error_handler: PipelineErrorHandler | None = None,
        queue_size: int = 16,
        max_workers: int = 4,
        thread_join_timeout: float = 2.0,
    ) -> None:
        self._error_handler = error_handler
        self._queue_size = queue_size
        self._thread_join_timeout = thread_join_timeout

        self._stage_queues: dict[str, Queue] = {}
        self._stage_threads: dict[str, threading.Thread] = {}
        self._stage_names: list[str] = []
        self._result_queue: Queue = Queue()
        self._running = False
        self._initialized = False

        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # PCF 1.3 -- thread-safe statistics
        self._stats_lock = threading.Lock()
        self._total_processed: int = 0
        self._frames_dropped: int = 0
        self._stage_times: dict[str, list[float]] = {}

        # Monotonic frame sequencing to discard stale out-of-order results
        self._frame_seq: int = 0
        self._highest_seen_seq: int = 0

    # -- public interface --------------------------------------------------

    def run_pipeline(
        self,
        stages: list[PipelineStageProtocol],
        initial_input: dict[str, Any],
    ) -> StageResult:
        """Submit *initial_input* and return the latest completed result.

        On the first call the worker threads are created lazily.  Each
        subsequent call enqueues a new frame for the first stage and
        drains any results that have completed the full pipeline.  If no
        result is ready yet the method returns a successful empty result
        (the frame is still in-flight).
        """
        if not self._initialized:
            self._setup_workers(stages)

        if self._stage_names:
            first_queue = self._stage_queues.get(self._stage_names[0])
            if first_queue is not None:
                data = dict(initial_input)
                data["_pipeline_start"] = time.perf_counter()
                self._frame_seq += 1
                data["_frame_seq"] = self._frame_seq
                try:
                    first_queue.put_nowait(data)
                except Full:
                    with self._stats_lock:
                        self._frames_dropped += 1
                    logger.debug(
                        "Frame %d dropped: queue '%s' full (size %d)",
                        self._frame_seq, self._stage_names[0], self._queue_size,
                    )

        latest: StageResult | None = None
        try:
            while True:
                result = self._result_queue.get_nowait()
                seq = result.data.get("_frame_seq", 0)
                if seq >= self._highest_seen_seq:
                    self._highest_seen_seq = seq
                    latest = result
                else:
                    logger.debug(
                        "Stale result discarded: seq %d < highest seen %d",
                        seq, self._highest_seen_seq,
                    )
        except Empty:
            pass

        if latest is not None:
            return latest
        return StageResult(success=True, data={}, duration_ms=0.0)

    def stop(self) -> None:
        """Stop all worker threads and release the executor.

        PCF 1.1 -- drain-and-retry loop guarantees every worker receives
        a poison pill even when its input queue is full.

        PCF 1.4 -- executor shutdown is bounded and non-blocking.
        """
        logger.debug("AsyncStrategy.stop() called, sending poison pills")
        self._running = False

        for name in list(self._stage_names):
            queue = self._stage_queues.get(name)
            if queue is None:
                continue
            for attempt in range(self._queue_size + 5):
                try:
                    queue.put(_POISON, timeout=0.1)
                    logger.debug(
                        "Poison pill delivered to '%s' on attempt %d", name, attempt + 1,
                    )
                    break
                except Full:
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

        for name, thread in self._stage_threads.items():
            thread.join(timeout=self._thread_join_timeout)
            if thread.is_alive():
                logger.debug(
                    "Worker '%s' still alive after %.1fs join timeout",
                    name, self._thread_join_timeout,
                )
            else:
                logger.debug("Worker '%s' joined successfully", name)

        self._executor.shutdown(wait=False, cancel_futures=True)

        self._stage_queues.clear()
        self._stage_threads.clear()
        self._stage_names.clear()
        self._initialized = False
        logger.debug("AsyncStrategy stopped and cleaned up")

    def cleanup(self) -> None:
        """Release all resources.  Safe to call multiple times."""
        if self._initialized:
            self.stop()

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of async-pipeline statistics."""
        with self._stats_lock:
            avg_times: dict[str, float] = {}
            for stage_name, times in self._stage_times.items():
                if times:
                    avg_times[stage_name] = sum(times) / len(times)
            return {
                "total_processed": self._total_processed,
                "frames_dropped": self._frames_dropped,
                "active_stages": len(self._stage_threads),
                "avg_stage_times_ms": avg_times,
                "queue_sizes": {
                    n: q.qsize() for n, q in self._stage_queues.items()
                },
            }

    # -- internals ---------------------------------------------------------

    def _setup_workers(self, stages: list[PipelineStageProtocol]) -> None:
        names: list[str] = []
        for i, stage in enumerate(stages):
            name = getattr(stage, "name", type(stage).__name__)
            while name in self._stage_queues:
                name = f"{name}_{i}"
            names.append(name)
            self._stage_queues[name] = Queue(maxsize=self._queue_size)

        self._stage_names = names
        self._running = True

        for i, stage in enumerate(stages):
            name = names[i]
            next_name = names[i + 1] if i + 1 < len(stages) else None
            thread_name = f"AsyncStrategy-{name}"
            thread = threading.Thread(
                target=self._stage_worker,
                args=(stage, name, next_name),
                daemon=True,
                name=thread_name,
            )
            self._stage_threads[name] = thread
            logger.debug(
                "Worker created: thread=%s  stage=%s  queue_size=%d  next=%s",
                thread_name, name, self._queue_size, next_name or "(result)",
            )
            thread.start()

        self._initialized = True
        logger.debug(
            "AsyncStrategy initialized: %d workers, stages=[%s]",
            len(names), ", ".join(names),
        )

    _MAX_TIMING_SAMPLES = 100
    _STATS_FLUSH_INTERVAL = 10

    def _stage_worker(
        self,
        stage: PipelineStageProtocol,
        stage_name: str,
        next_stage_name: str | None,
    ) -> None:
        input_queue = self._stage_queues[stage_name]
        is_last_stage = next_stage_name is None

        local_times: list[float] = []
        local_completed = 0

        logger.debug("Worker '%s' started, waiting for frames", stage_name)

        while self._running:
            try:
                data = input_queue.get(timeout=0.1)
            except Empty:
                if local_times:
                    self._flush_stage_stats(stage_name, local_times, local_completed)
                    local_times = []
                    local_completed = 0
                continue

            if data is _POISON:
                logger.debug("Worker '%s' received poison pill, exiting", stage_name)
                break

            frame_seq = data.get("_frame_seq", "?")
            logger.debug("Worker '%s' processing frame %s", stage_name, frame_seq)

            start = time.perf_counter()
            try:
                result = stage.execute(data)
            except Exception as exc:
                self._record_error(stage_name, exc, ErrorSeverity.HIGH)
                self._result_queue.put(
                    StageResult(success=False, error=f"{stage_name}: {exc}")
                )
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000
            local_times.append(elapsed_ms)

            if not result.success:
                self._record_error(
                    stage_name,
                    RuntimeError(result.error or "stage returned failure"),
                    ErrorSeverity.MEDIUM,
                )
                self._result_queue.put(result)
                continue

            data.update(result.data)

            if not is_last_stage:
                output_queue = self._stage_queues.get(next_stage_name)
                if output_queue is not None:
                    try:
                        output_queue.put(dict(data), timeout=0.5)
                        logger.debug(
                            "Worker '%s' frame %s -> queue '%s' (%.1fms)",
                            stage_name, frame_seq, next_stage_name, elapsed_ms,
                        )
                    except Full:
                        with self._stats_lock:
                            self._frames_dropped += 1
                        logger.debug(
                            "Worker '%s' frame %s dropped: queue '%s' full",
                            stage_name, frame_seq, next_stage_name,
                        )
            else:
                local_completed += 1
                final_data = dict(data)
                pipeline_start = final_data.pop("_pipeline_start", None)
                final_data.pop("_frame_seq", None)
                total_ms = (
                    (time.perf_counter() - pipeline_start) * 1000
                    if pipeline_start is not None
                    else elapsed_ms
                )
                self._result_queue.put(
                    StageResult(success=True, data=final_data, duration_ms=total_ms)
                )
                logger.debug(
                    "Worker '%s' frame %s -> result queue (stage %.1fms, total %.1fms)",
                    stage_name, frame_seq, elapsed_ms, total_ms,
                )

            if len(local_times) >= self._STATS_FLUSH_INTERVAL:
                self._flush_stage_stats(stage_name, local_times, local_completed)
                self._log_queue_sizes()
                local_times = []
                local_completed = 0

        if local_times:
            self._flush_stage_stats(stage_name, local_times, local_completed)

        logger.debug("Worker '%s' exited", stage_name)

    def _flush_stage_stats(
        self,
        stage_name: str,
        times: list[float],
        completed: int,
    ) -> None:
        """Batch-write accumulated per-stage timing and frame counts."""
        with self._stats_lock:
            stored = self._stage_times.setdefault(stage_name, [])
            stored.extend(times)
            if len(stored) > self._MAX_TIMING_SAMPLES:
                del stored[:len(stored) - self._MAX_TIMING_SAMPLES]
            if completed:
                self._total_processed += completed

    def _log_queue_sizes(self) -> None:
        """Periodic snapshot of all stage queue depths."""
        sizes = {n: q.qsize() for n, q in self._stage_queues.items()}
        with self._stats_lock:
            total = self._total_processed
            dropped = self._frames_dropped
        logger.debug(
            "Queue snapshot  queues=%s  processed=%d  dropped=%d",
            sizes, total, dropped,
        )

    def _record_error(
        self,
        component: str,
        error: Exception,
        severity: ErrorSeverity,
    ) -> None:
        if self._error_handler:
            self._error_handler.handle_error(
                component=component, error=error, severity=severity,
            )
        else:
            logger.error("Stage %s failed: %s", component, error)


# ---------------------------------------------------------------------------
# CustomStrategy
# ---------------------------------------------------------------------------

class CustomStrategy:
    """Per-stage execution-mode selection.

    Each stage can be configured to run either sequentially (in the
    caller's thread) or asynchronously (offloaded to a bounded thread
    pool with a per-stage timeout).  Stages default to sequential when
    not explicitly configured.

    The thread pool uses bounded shutdown (``cancel_futures=True``).
    """

    _MAX_TIMING_SAMPLES = 100

    def __init__(
        self,
        stage_modes: dict[str, ExecutionMode] | None = None,
        error_handler: PipelineErrorHandler | None = None,
        async_timeout: float = 5.0,
        max_workers: int = 4,
    ) -> None:
        self._stage_modes = stage_modes or {}
        self._error_handler = error_handler
        self._async_timeout = async_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stats_lock = threading.Lock()
        self._stage_times: dict[str, list[float]] = {}

    def run_pipeline(
        self,
        stages: list[PipelineStageProtocol],
        initial_input: dict[str, Any],
    ) -> StageResult:
        current_data = dict(initial_input)
        last_result = StageResult(success=True, data=current_data)

        for stage in stages:
            stage_name = getattr(stage, "name", type(stage).__name__)
            mode = self._stage_modes.get(stage_name, ExecutionMode.SEQUENTIAL)

            logger.debug(
                "CustomStrategy running '%s' in %s mode", stage_name, mode.name,
            )
            stage_start = time.perf_counter()

            try:
                if mode == ExecutionMode.ASYNC:
                    future = self._executor.submit(stage.execute, current_data)
                    try:
                        result = future.result(timeout=self._async_timeout)
                    except Exception:
                        future.cancel()
                        raise
                else:
                    result = stage.execute(current_data)
            except Exception as exc:
                self._record_error(stage_name, exc, ErrorSeverity.HIGH)
                return StageResult(
                    success=False,
                    data=current_data,
                    duration_ms=last_result.duration_ms,
                    error=f"{stage_name}: {exc}",
                )

            stage_elapsed = (time.perf_counter() - stage_start) * 1000

            if not result.success:
                self._record_error(
                    stage_name,
                    RuntimeError(result.error or "stage returned failure"),
                    ErrorSeverity.MEDIUM,
                )
                logger.debug(
                    "CustomStrategy '%s' (%s) failed after %.1fms",
                    stage_name, mode.name, stage_elapsed,
                )
                return result

            logger.debug(
                "CustomStrategy '%s' (%s) completed in %.1fms",
                stage_name, mode.name, stage_elapsed,
            )

            with self._stats_lock:
                times = self._stage_times.setdefault(stage_name, [])
                times.append(result.duration_ms)
                if len(times) > self._MAX_TIMING_SAMPLES:
                    del times[:len(times) - self._MAX_TIMING_SAMPLES]

            current_data.update(result.data)
            last_result = StageResult(
                success=True,
                data=current_data,
                duration_ms=last_result.duration_ms + result.duration_ms,
            )

        return last_result

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of custom strategy statistics."""
        with self._stats_lock:
            avg_times: dict[str, float] = {}
            for stage_name, times in self._stage_times.items():
                if times:
                    avg_times[stage_name] = sum(times) / len(times)
            return {
                'avg_stage_times_ms': avg_times,
            }

    def stop(self) -> None:
        """Shut down the thread pool (bounded, non-blocking)."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def cleanup(self) -> None:
        """Release resources.  Safe to call multiple times."""
        self.stop()

    def _record_error(
        self,
        component: str,
        error: Exception,
        severity: ErrorSeverity,
    ) -> None:
        if self._error_handler:
            self._error_handler.handle_error(
                component=component, error=error, severity=severity,
            )
        else:
            logger.error("Stage %s failed: %s", component, error)


# ---------------------------------------------------------------------------
# SubprocessStrategy
# ---------------------------------------------------------------------------

class SubprocessStrategy:
    """Execute stages in isolated sub-processes for crash resilience.

    If a ``SubprocessManager`` is provided, stages whose names contain a
    known keyword (``capture``, ``ocr``, ``translat``) are routed to the
    corresponding subprocess.  Stages without a matching subprocess, or
    when no manager is configured, fall back to in-process sequential
    execution.
    """

    _STAGE_SUBPROCESS_MAP = {
        "capture": "capture_subprocess",
        "ocr": "ocr_subprocess",
        "translat": "translation_subprocess",
    }

    def __init__(
        self,
        error_handler: PipelineErrorHandler | None = None,
        subprocess_manager: Any = None,
        process_timeout: float = 5.0,
    ) -> None:
        self._error_handler = error_handler
        self._subprocess_manager = subprocess_manager
        self._process_timeout = process_timeout
        self._fallback = SequentialStrategy(error_handler=error_handler)

    def run_pipeline(
        self,
        stages: list[PipelineStageProtocol],
        initial_input: dict[str, Any],
    ) -> StageResult:
        if self._subprocess_manager is None:
            return self._fallback.run_pipeline(stages, initial_input)

        current_data = dict(initial_input)
        last_result = StageResult(success=True, data=current_data)

        for stage in stages:
            stage_name = getattr(stage, "name", type(stage).__name__)
            try:
                result = self._run_stage(stage, stage_name, current_data)
            except Exception as exc:
                self._record_error(stage_name, exc, ErrorSeverity.HIGH)
                return StageResult(
                    success=False,
                    data=current_data,
                    duration_ms=last_result.duration_ms,
                    error=f"{stage_name}: {exc}",
                )

            if not result.success:
                self._record_error(
                    stage_name,
                    RuntimeError(result.error or "stage returned failure"),
                    ErrorSeverity.MEDIUM,
                )
                return result

            current_data.update(result.data)
            last_result = StageResult(
                success=True,
                data=dict(current_data),
                duration_ms=last_result.duration_ms + result.duration_ms,
            )

        return last_result

    def _run_stage(
        self,
        stage: PipelineStageProtocol,
        name: str,
        data: dict[str, Any],
    ) -> StageResult:
        subprocess = self._resolve_subprocess(name)
        if subprocess is not None and getattr(subprocess, "is_alive", lambda: False)():
            start = time.perf_counter()
            result_data = subprocess.process_data(data, timeout=self._process_timeout)
            elapsed = (time.perf_counter() - start) * 1000
            if result_data is not None:
                return StageResult(success=True, data=result_data, duration_ms=elapsed)
            return StageResult(
                success=False,
                error=f"{name}: subprocess returned no data",
                duration_ms=elapsed,
            )
        return stage.execute(data)

    def _resolve_subprocess(self, stage_name: str) -> Any:
        name_lower = stage_name.lower()
        for keyword, attr in self._STAGE_SUBPROCESS_MAP.items():
            if keyword in name_lower:
                return getattr(self._subprocess_manager, attr, None)
        return None

    def _record_error(
        self,
        component: str,
        error: Exception,
        severity: ErrorSeverity,
    ) -> None:
        if self._error_handler:
            self._error_handler.handle_error(
                component=component, error=error, severity=severity,
            )
        else:
            logger.error("Stage %s failed: %s", component, error)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (removed in Phase 2A.5)
# ---------------------------------------------------------------------------

OptimizedStrategy = CustomStrategy
