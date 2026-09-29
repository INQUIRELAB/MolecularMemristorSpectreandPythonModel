import os
import sys
import time
import threading
import math
import uuid
import numpy as np
import concurrent.futures
from .sys_utils import get_temp_history_dir

class SSDPipelineTelemetry:
    """
    High-resolution telemetry tracker for the multi-tier SSD streaming history pipeline.
    Captures microsecond timings for PCIe DMA copies, thread dispatch latency, memory-map memcpy,
    file growth truncation, and NVMe barrier sync times.
    """
    _instance = None

    def __init__(self):
        self._lock = threading.Lock()
        self.total_steps = 0
        self.total_bytes = 0
        self.total_chunks = 0
        self.dma_times = []
        self.dma_bytes = []
        self.dispatch_latencies = []
        self.write_times = []
        self.write_bytes = []
        self.expansion_times = []
        self.flush_wait_times = []
        self.channel_times = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SSDPipelineTelemetry()
        return cls._instance

    def record_chunk(self, completed_steps, total_bytes, dispatch_latency, write_duration, channel_breakdown=None):
        with self._lock:
            self.total_steps += completed_steps
            self.total_bytes += total_bytes
            self.total_chunks += 1
            if dispatch_latency > 0:
                self.dispatch_latencies.append(dispatch_latency)
            if write_duration > 0:
                self.write_times.append(write_duration)
                self.write_bytes.append(total_bytes)
            if channel_breakdown:
                for ch, dur in channel_breakdown.items():
                    self.channel_times[ch] = self.channel_times.get(ch, 0.0) + dur

    def record_dma(self, duration_sec, bytes_count):
        with self._lock:
            self.dma_times.append(duration_sec)
            self.dma_bytes.append(bytes_count)

    def record_expansion(self, duration_sec):
        with self._lock:
            self.expansion_times.append(duration_sec)

    def record_flush_wait(self, duration_sec):
        with self._lock:
            self.flush_wait_times.append(duration_sec)

    def reset(self):
        with self._lock:
            self.total_steps = 0
            self.total_bytes = 0
            self.total_chunks = 0
            self.dma_times.clear()
            self.dma_bytes.clear()
            self.dispatch_latencies.clear()
            self.write_times.clear()
            self.write_bytes.clear()
            self.expansion_times.clear()
            self.flush_wait_times.clear()
            self.channel_times.clear()


def generate_ssd_pipeline_report(sim_dir):
    """
    Generates a high-precision, formatted SSD pipeline execution summary report
    in the simulation_profiles/ directory when profiling is enabled.
    """
    telemetry = SSDPipelineTelemetry.get_instance()
    if telemetry.total_chunks == 0 and len(telemetry.dma_times) == 0:
        return
    
    total_steps = telemetry.total_steps
    total_mb = telemetry.total_bytes / (1024 * 1024)
    total_chunks = telemetry.total_chunks
    
    dma_hits = len(telemetry.dma_times)
    dma_total_time = sum(telemetry.dma_times)
    dma_avg_time = (dma_total_time / dma_hits) if dma_hits > 0 else 0.0
    dma_total_mb = sum(telemetry.dma_bytes) / (1024 * 1024)
    dma_bw_gb_s = (dma_total_mb / (1024 * dma_total_time)) if dma_total_time > 0 else 0.0
    
    disp_hits = len(telemetry.dispatch_latencies)
    disp_total_time = sum(telemetry.dispatch_latencies)
    disp_avg_time = (disp_total_time / disp_hits) if disp_hits > 0 else 0.0
    
    wr_hits = len(telemetry.write_times)
    wr_total_time = sum(telemetry.write_times)
    wr_avg_time = (wr_total_time / wr_hits) if wr_hits > 0 else 0.0
    wr_total_mb = sum(telemetry.write_bytes) / (1024 * 1024)
    wr_bw_gb_s = (wr_total_mb / (1024 * wr_total_time)) if wr_total_time > 0 else 0.0
    
    exp_hits = len(telemetry.expansion_times)
    exp_total_time = sum(telemetry.expansion_times)
    exp_avg_time = (exp_total_time / exp_hits) if exp_hits > 0 else 0.0
    
    fl_hits = len(telemetry.flush_wait_times)
    fl_total_time = sum(telemetry.flush_wait_times)
    fl_avg_time = (fl_total_time / fl_hits) if fl_hits > 0 else 0.0
    fl_bw_mb_s = (total_mb / fl_total_time) if fl_total_time > 0 else 0.0
    
    pipeline_total_time = dma_total_time + disp_total_time + wr_total_time + exp_total_time + fl_total_time
    
    def fmt_time(t_sec):
        if t_sec >= 1.0:
            return f"{t_sec:.3f} s"
        elif t_sec >= 1e-3:
            return f"{t_sec * 1000.0:.2f} ms"
        elif t_sec >= 1e-6:
            return f"{t_sec * 1e6:.1f} us"
        else:
            return f"{t_sec * 1e9:.1f} ns"

    lines = []
    lines.append("=" * 110)
    lines.append("                      MOLMEM SSD PIPELINE GRANULAR EXECUTION REPORT")
    lines.append("=" * 110)
    lines.append(f"Total Simulation Steps Recorded : {total_steps:,} Steps")
    lines.append(f"Total Data Streamed to NVMe     : {total_mb:.2f} MB (across on-disk memory-mapped channels)")
    lines.append(f"Total Chunks Dispatched         : {total_chunks:,} Chunks")
    lines.append("-" * 110)
    lines.append(f"{'Pipeline Stage':<40} | {'Hits':<6} | {'Total Time':<14} | {'Avg / Chunk':<14} | {'Bandwidth / Throughput':<22}")
    lines.append("-" * 110)
    
    if dma_hits > 0:
        dma_bw_str = f"{dma_bw_gb_s:.2f} GB/s (PCIe)"
        lines.append(f"{'1. GPU->Host PCIe DMA Transfer':<40} | {dma_hits:<6} | {fmt_time(dma_total_time):<14} | {fmt_time(dma_avg_time):<14} | {dma_bw_str:<22}")
    else:
        lines.append(f"{'1. GPU->Host PCIe DMA Transfer':<40} | {'0':<6} | {'N/A (CPU Mode)':<14} | {'N/A':<14} | {'N/A':<22}")
        
    lines.append(f"{'2. Thread Queue Dispatch Latency':<40} | {disp_hits:<6} | {fmt_time(disp_total_time):<14} | {fmt_time(disp_avg_time):<14} | {'-':<22}")
    lines.append(f"{'3. Memory-Map Channel Writes (memcpy)':<40} | {wr_hits:<6} | {fmt_time(wr_total_time):<14} | {fmt_time(wr_avg_time):<14} | {f'{wr_bw_gb_s:.2f} GB/s (RAM)':<22}")
    lines.append(f"{'4. File Capacity Expansion (Truncate)':<40} | {exp_hits:<6} | {fmt_time(exp_total_time):<14} | {fmt_time(exp_avg_time):<14} | {'-':<22}")
    lines.append(f"{'5. Main-Thread Barrier Sync & Flush':<40} | {fl_hits:<6} | {fmt_time(fl_total_time):<14} | {fmt_time(fl_avg_time):<14} | {f'{fl_bw_mb_s:.1f} MB/s (NVMe)':<22}")
    lines.append("-" * 110)
    lines.append(f"{'OVERALL SSD PIPELINE ACTIVE OVERHEAD':<40} | {total_chunks:<6} | {fmt_time(pipeline_total_time):<14} | {fmt_time(pipeline_total_time / max(1, total_chunks)):<14} | {'Fully Overlapped':<22}")
    lines.append("=" * 110)
    
    if telemetry.channel_times:
        lines.append("\n  INDIVIDUAL CHANNEL WRITE BREAKDOWN:")
        lines.append("  " + "-" * 70)
        lines.append(f"  {'Channel Name':<25} | {'Total Time Spent Writing':<25}")
        lines.append("  " + "-" * 70)
        for ch_name, ch_time in sorted(telemetry.channel_times.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {ch_name:<25} | {fmt_time(ch_time):<25}")
        lines.append("  " + "-" * 70)

    report_content = "\n".join(lines) + "\n"
    os.makedirs(sim_dir, exist_ok=True)
    report_path = os.path.join(sim_dir, "ssd_pipeline_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)


def _disable_trace_for_writer_thread():
    import sys, threading
    try:
        sys.settrace(None)
        threading.settrace(None)
    except Exception:
        pass


class ThreadPoolPingPongManager:
    """
    Manages dedicated ping-pong double-buffer sets and asynchronous disk writer threads.
    Sizes the worker pool dynamically to min(max(2, num_instances * 2), int(0.8 * cpu_count)).
    Eliminates thread starvation for batch runs while avoiding multi-thread GIL lock storms on single runs.
    """
    _instance = None

    def __init__(self, target_ratio=0.8):
        self.target_ratio = target_ratio
        self.max_hw_workers = max(1, int(target_ratio * (os.cpu_count() or 1)))
        self._active_sim_count = 0
        self._buffers = {}
        self._lock = threading.Lock()
        self.num_workers = min(2, self.max_hw_workers)
        self._writer_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="MolmemHistWriter",
            initializer=_disable_trace_for_writer_thread
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ThreadPoolPingPongManager()
        return cls._instance

    def register_instance(self, count=1):
        with self._lock:
            self._active_sim_count += count
            desired_workers = min(max(2, self._active_sim_count * 2), self.max_hw_workers)
            if desired_workers > self.num_workers:
                self.num_workers = desired_workers
                old_pool = self._writer_pool
                self._writer_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.num_workers,
                    thread_name_prefix="MolmemHistWriter",
                    initializer=_disable_trace_for_writer_thread
                )
                if old_pool is not None:
                    try:
                        old_pool.shutdown(wait=False)
                    except Exception:
                        pass

    def unregister_instance(self, count=1):
        with self._lock:
            self._active_sim_count = max(0, self._active_sim_count - count)

    def get_worker_buffers(self, worker_id, num_nodes, num_devices, hist_capacity, recordHistory, recordStepTiming, batch_size=1):
        with self._lock:
            existing = self._buffers.get(worker_id)
            
            need_realloc = False
            if existing is None:
                need_realloc = True
            else:
                if existing.get('capacity', 0) < hist_capacity:
                    need_realloc = True
                elif existing.get('num_nodes') != num_nodes or existing.get('num_devices') != num_devices:
                    need_realloc = True
                elif existing.get('recordHistory') != recordHistory or existing.get('recordStepTiming') != recordStepTiming:
                    need_realloc = True
                elif existing.get('batch_size', 1) != batch_size:
                    need_realloc = True
                    
            if need_realloc:
                alloc_cap = hist_capacity
                if batch_size > 1:
                    tArr_buffers = [np.zeros((batch_size, alloc_cap), dtype=np.float64), np.zeros((batch_size, alloc_cap), dtype=np.float64)]
                    stepTimeHist_buffers = [
                        np.zeros((batch_size, alloc_cap), dtype=np.float64) if recordStepTiming else np.zeros((batch_size, 2), dtype=np.float64),
                        np.zeros((batch_size, alloc_cap), dtype=np.float64) if recordStepTiming else np.zeros((batch_size, 2), dtype=np.float64)
                    ]
                    vHist_buffers = [
                        np.zeros((batch_size, num_nodes, alloc_cap), dtype=np.float64) if (recordHistory & 1) else np.zeros((batch_size, num_nodes, 2), dtype=np.float64),
                        np.zeros((batch_size, num_nodes, alloc_cap), dtype=np.float64) if (recordHistory & 1) else np.zeros((batch_size, num_nodes, 2), dtype=np.float64)
                    ]
                    if num_devices > 0:
                        iHist_buffers = [
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 2) else np.zeros((batch_size, num_devices, 2), dtype=np.float64),
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 2) else np.zeros((batch_size, num_devices, 2), dtype=np.float64)
                        ]
                        stateHist_buffers = [
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 4) else np.zeros((batch_size, num_devices, 2), dtype=np.float64),
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 4) else np.zeros((batch_size, num_devices, 2), dtype=np.float64)
                        ]
                        f22Hist_buffers = [
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 8) else np.zeros((batch_size, num_devices, 2), dtype=np.float64),
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 8) else np.zeros((batch_size, num_devices, 2), dtype=np.float64)
                        ]
                        gHist_buffers = [
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 16) else np.zeros((batch_size, num_devices, 2), dtype=np.float64),
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 16) else np.zeros((batch_size, num_devices, 2), dtype=np.float64)
                        ]
                        tempHist_buffers = [
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 32) else np.zeros((batch_size, num_devices, 2), dtype=np.float64),
                            np.zeros((batch_size, num_devices, alloc_cap), dtype=np.float64) if (recordHistory & 32) else np.zeros((batch_size, num_devices, 2), dtype=np.float64)
                        ]
                    else:
                        iHist_buffers = [np.zeros((batch_size, 0, 2), dtype=np.float64), np.zeros((batch_size, 0, 2), dtype=np.float64)]
                        stateHist_buffers = [np.zeros((batch_size, 0, 2), dtype=np.float64), np.zeros((batch_size, 0, 2), dtype=np.float64)]
                        f22Hist_buffers = [np.zeros((batch_size, 0, 2), dtype=np.float64), np.zeros((batch_size, 0, 2), dtype=np.float64)]
                        gHist_buffers = [np.zeros((batch_size, 0, 2), dtype=np.float64), np.zeros((batch_size, 0, 2), dtype=np.float64)]
                        tempHist_buffers = [np.zeros((batch_size, 0, 2), dtype=np.float64), np.zeros((batch_size, 0, 2), dtype=np.float64)]
                else:
                    tArr_buffers = [np.zeros(alloc_cap, dtype=np.float64), np.zeros(alloc_cap, dtype=np.float64)]
                    stepTimeHist_buffers = [
                        np.zeros(alloc_cap, dtype=np.float64) if recordStepTiming else np.zeros(2, dtype=np.float64),
                        np.zeros(alloc_cap, dtype=np.float64) if recordStepTiming else np.zeros(2, dtype=np.float64)
                    ]
                    vHist_buffers = [
                        np.zeros((num_nodes, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 1) else np.zeros((num_nodes, 2), dtype=np.float64, order='F'),
                        np.zeros((num_nodes, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 1) else np.zeros((num_nodes, 2), dtype=np.float64, order='F')
                    ]
                    if num_devices > 0:
                        iHist_buffers = [
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 2) else np.zeros((num_devices, 2), dtype=np.float64, order='F'),
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 2) else np.zeros((num_devices, 2), dtype=np.float64, order='F')
                        ]
                        stateHist_buffers = [
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 4) else np.zeros((num_devices, 2), dtype=np.float64, order='F'),
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 4) else np.zeros((num_devices, 2), dtype=np.float64, order='F')
                        ]
                        f22Hist_buffers = [
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 8) else np.zeros((num_devices, 2), dtype=np.float64, order='F'),
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 8) else np.zeros((num_devices, 2), dtype=np.float64, order='F')
                        ]
                        gHist_buffers = [
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 16) else np.zeros((num_devices, 2), dtype=np.float64, order='F'),
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 16) else np.zeros((num_devices, 2), dtype=np.float64, order='F')
                        ]
                        tempHist_buffers = [
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 32) else np.zeros((num_devices, 2), dtype=np.float64, order='F'),
                            np.zeros((num_devices, alloc_cap), dtype=np.float64, order='F') if (recordHistory & 32) else np.zeros((num_devices, 2), dtype=np.float64, order='F')
                        ]
                    else:
                        iHist_buffers = [np.zeros((0, 2), dtype=np.float64, order='F'), np.zeros((0, 2), dtype=np.float64, order='F')]
                        stateHist_buffers = [np.zeros((0, 2), dtype=np.float64, order='F'), np.zeros((0, 2), dtype=np.float64, order='F')]
                        f22Hist_buffers = [np.zeros((0, 2), dtype=np.float64, order='F'), np.zeros((0, 2), dtype=np.float64, order='F')]
                        gHist_buffers = [np.zeros((0, 2), dtype=np.float64, order='F'), np.zeros((0, 2), dtype=np.float64, order='F')]
                        tempHist_buffers = [np.zeros((0, 2), dtype=np.float64, order='F'), np.zeros((0, 2), dtype=np.float64, order='F')]

                existing = {
                    'capacity': alloc_cap,
                    'num_nodes': num_nodes,
                    'num_devices': num_devices,
                    'batch_size': batch_size,
                    'recordHistory': recordHistory,
                    'recordStepTiming': recordStepTiming,
                    'tArr_buffers': tArr_buffers,
                    'stepTimeHist_buffers': stepTimeHist_buffers,
                    'vHist_buffers': vHist_buffers,
                    'iHist_buffers': iHist_buffers,
                    'stateHist_buffers': stateHist_buffers,
                    'f22Hist_buffers': f22Hist_buffers,
                    'gHist_buffers': gHist_buffers,
                    'tempHist_buffers': tempHist_buffers
                }
                self._buffers[worker_id] = existing
            return existing

class DiskHistoryStore:
    """
    High-performance on-disk memory-mapped array manager for transient simulation history.
    Stores tArr, vHist, iHist, stateHist, f22Hist, gHist, tempHist directly in binary
    files via numpy.memmap, eliminating RAM growth and garbage-collection stalls.
    Supports asynchronous background chunk writes with double buffering.
    Supports arbitrary batch dimensions (batch_size >= 1).
    """
    def __init__(self, num_nodes, num_devices=0, initial_capacity=1000, sim_id=None,
                 v_flag=True, i_flag=True, state_flag=True, f22_flag=True, g_flag=True,
                 temp_flag=True, step_time_flag=False, batch_size=1):
        self.pid = os.getpid()
        self.sim_id = sim_id if sim_id is not None else str(uuid.uuid4())[:8]
        self.num_nodes = max(1, num_nodes)
        self.num_devices = max(0, num_devices)
        self.batch_size = max(1, int(batch_size))
        
        # Budget initial capacity dynamically so on-disk files start at <= 10MB total
        dev_footprint = max(1, self.num_devices, self.num_nodes) * self.batch_size
        max_init_from_budget = max(100, 1250000 // dev_footprint)
        self.capacity = max(100, min(initial_capacity if initial_capacity > 0 else 1000, max_init_from_budget))
        self.active_len = 0
        self.temp_dir = get_temp_history_dir(auto_create=True)
        
        self.v_flag = v_flag
        self.i_flag = i_flag
        self.state_flag = state_flag
        self.f22_flag = f22_flag
        self.g_flag = g_flag
        self.temp_flag = temp_flag
        self.step_time_flag = step_time_flag
        
        self._files = {}
        self._maps = {}
        self._shapes = {}
        self._orders = {}
        self._lock_fds = {}
        self._cleaned = False
        self._writer_futures = []
        ThreadPoolPingPongManager.get_instance().register_instance(1)
        
        self._init_channels()

    def _get_filename(self, channel_name):
        return os.path.join(self.temp_dir, f"hist_pid{self.pid}_{self.sim_id}_{channel_name}.dat")

    def _init_channels(self):
        if self.batch_size > 1:
            # Batched 3D / 2D Channels
            self._create_channel('tArr', (self.batch_size, self.capacity), order='C')
            if self.step_time_flag:
                self._create_channel('stepTime', (self.batch_size, self.capacity), order='C')
            else:
                self._maps['stepTime'] = np.zeros((self.batch_size, 2), dtype=np.float64)
                
            if self.v_flag:
                self._create_channel('vHist', (self.batch_size, self.num_nodes, self.capacity), order='C')
            else:
                self._maps['vHist'] = np.zeros((self.batch_size, self.num_nodes, 2), dtype=np.float64)
                
            if self.num_devices > 0:
                if self.i_flag:
                    self._create_channel('iHist', (self.batch_size, self.num_devices, self.capacity), order='C')
                else:
                    self._maps['iHist'] = np.zeros((self.batch_size, self.num_devices, 2), dtype=np.float64)
                    
                if self.state_flag:
                    self._create_channel('stateHist', (self.batch_size, self.num_devices, self.capacity), order='C')
                else:
                    self._maps['stateHist'] = np.zeros((self.batch_size, self.num_devices, 2), dtype=np.float64)
                    
                if self.f22_flag:
                    self._create_channel('f22Hist', (self.batch_size, self.num_devices, self.capacity), order='C')
                else:
                    self._maps['f22Hist'] = np.zeros((self.batch_size, self.num_devices, 2), dtype=np.float64)
                    
                if self.g_flag:
                    self._create_channel('gHist', (self.batch_size, self.num_devices, self.capacity), order='C')
                else:
                    self._maps['gHist'] = np.zeros((self.batch_size, self.num_devices, 2), dtype=np.float64)
                    
                if self.temp_flag:
                    self._create_channel('tempHist', (self.batch_size, self.num_devices, self.capacity), order='C')
                else:
                    self._maps['tempHist'] = np.zeros((self.batch_size, self.num_devices, 2), dtype=np.float64)
            else:
                dummy = np.zeros((self.batch_size, 0, 2), dtype=np.float64)
                for dev_ch in ['iHist', 'stateHist', 'f22Hist', 'gHist', 'tempHist']:
                    self._maps[dev_ch] = dummy
        else:
            # 1. tArr: shape (capacity,)
            self._create_channel('tArr', (self.capacity,), order='C')
            
            # 2. stepTime: shape (capacity,)
            if self.step_time_flag:
                self._create_channel('stepTime', (self.capacity,), order='C')
            else:
                self._maps['stepTime'] = np.zeros(2, dtype=np.float64)
                
            # 3. vHist: shape (num_nodes, capacity)
            if self.v_flag:
                self._create_channel('vHist', (self.num_nodes, self.capacity), order='F')
            else:
                self._maps['vHist'] = np.zeros((self.num_nodes, 2), dtype=np.float64, order='F')
                
            # 4. Device Channels: shape (num_devices, capacity)
            if self.num_devices > 0:
                if self.i_flag:
                    self._create_channel('iHist', (self.num_devices, self.capacity), order='F')
                else:
                    self._maps['iHist'] = np.zeros((self.num_devices, 2), dtype=np.float64, order='F')
                    
                if self.state_flag:
                    self._create_channel('stateHist', (self.num_devices, self.capacity), order='F')
                else:
                    self._maps['stateHist'] = np.zeros((self.num_devices, 2), dtype=np.float64, order='F')
                    
                if self.f22_flag:
                    self._create_channel('f22Hist', (self.num_devices, self.capacity), order='F')
                else:
                    self._maps['f22Hist'] = np.zeros((self.num_devices, 2), dtype=np.float64, order='F')
                    
                if self.g_flag:
                    self._create_channel('gHist', (self.num_devices, self.capacity), order='F')
                else:
                    self._maps['gHist'] = np.zeros((self.num_devices, 2), dtype=np.float64, order='F')
                    
                if self.temp_flag:
                    self._create_channel('tempHist', (self.num_devices, self.capacity), order='F')
                else:
                    self._maps['tempHist'] = np.zeros((self.num_devices, 2), dtype=np.float64, order='F')
            else:
                dummy = np.zeros((0, 2), dtype=np.float64, order='F')
                for dev_ch in ['iHist', 'stateHist', 'f22Hist', 'gHist', 'tempHist']:
                    self._maps[dev_ch] = dummy

    def _create_channel(self, name, shape, order='F'):
        filepath = self._get_filename(name)
        self._files[name] = filepath
        self._shapes[name] = shape
        self._orders[name] = order
        
        m = np.memmap(filepath, dtype=np.float64, mode='w+', shape=shape, order=order)
        self._maps[name] = m
        if sys.platform != 'win32':
            try:
                import fcntl
                fd = os.open(filepath, os.O_RDONLY)
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                self._lock_fds[name] = fd
            except Exception:
                pass

    def ensure_capacity(self, required_capacity):
        if required_capacity <= self.capacity:
            return
        # Synchronously wait for any in-flight background writes before unmapping and expanding files
        self.wait_for_writes()
        
        t0_exp = time.perf_counter()
        dev_footprint = max(1, self.num_devices, self.num_nodes) * self.batch_size
        growth_chunk = max(10000, 5000000 // dev_footprint)
        new_capacity = max(required_capacity, self.capacity + growth_chunk, int(self.capacity * 2.0))
        
        for name in list(self._files.keys()):
            filepath = self._files[name]
            old_shape = self._shapes[name]
            order = self._orders[name]
            
            if len(old_shape) == 1:
                new_shape = (new_capacity,)
            elif len(old_shape) == 2:
                new_shape = (old_shape[0], new_capacity)
            elif len(old_shape) == 3:
                new_shape = (old_shape[0], old_shape[1], new_capacity)
            else:
                new_shape = old_shape[:-1] + (new_capacity,)
                
            self._shapes[name] = new_shape
            
            m = self._maps.get(name)
            if m is not None and isinstance(m, np.memmap):
                try:
                    m.flush()
                except Exception:
                    pass
                try:
                    if hasattr(m, '_mmap') and m._mmap is not None:
                        m._mmap.close()
                except Exception:
                    pass
                self._maps[name] = None
                
            num_elements = int(np.prod(new_shape))
            num_bytes = num_elements * 8
            with open(filepath, 'r+b') as f:
                f.truncate(num_bytes)
                
            new_m = np.memmap(filepath, dtype=np.float64, mode='r+', shape=new_shape, order=order)
            self._maps[name] = new_m
            
        self.capacity = new_capacity
        SSDPipelineTelemetry.get_instance().record_expansion(time.perf_counter() - t0_exp)

    @property
    def tArr(self):
        return self._maps['tArr']

    @property
    def stepTime(self):
        return self._maps['stepTime']

    @property
    def vHist(self):
        return self._maps['vHist']

    @property
    def iHist(self):
        return self._maps['iHist']

    @property
    def stateHist(self):
        return self._maps['stateHist']

    @property
    def f22Hist(self):
        return self._maps['f22Hist']

    @property
    def gHist(self):
        return self._maps['gHist']

    @property
    def tempHist(self):
        return self._maps['tempHist']

    def write_chunk_async(self, host_start, completed_steps, recordHistory, recordStepTiming,
                          tArr, vHist, iHist, stateHist, f22Hist, gHist, tempHist, stepTimeHist=None):
        """
        Asynchronously writes a completed simulation chunk to disk via a background thread pool,
        completely isolating the CPU ODE solver from OS page-faults and disk writeback latency.
        Zero copying occurs on the main thread when double-buffering ping-pong buffers are passed.
        """
        host_end = host_start + completed_steps
        self.ensure_capacity(host_end)
        submit_time = time.perf_counter()
        
        if recordHistory is True or recordHistory == 63:
            bitmask = 63
        elif isinstance(recordHistory, int) and not isinstance(recordHistory, bool):
            bitmask = recordHistory
        elif not recordHistory:
            bitmask = 0
        else:
            from .sys_utils import resolve_record_history_flags
            _, _, _, _, _, _, bitmask = resolve_record_history_flags(recordHistory)

        num_channels = 1 + (self.num_nodes if (bitmask & 1) else 0) + (self.num_devices * 5 if self.num_devices > 0 else 0) + (1 if recordStepTiming else 0)
        chunk_bytes = num_channels * completed_steps * 8
        
        def _do_write(h_start, h_end, n_steps, r_hist, r_time, t_src, v_src, i_src, s_src, f_src, g_src, temp_src, st_src, t_submit, n_bytes):
            t_start_work = time.perf_counter()
            disp_lat = t_start_work - t_submit
            ch_breakdown = {}
            is_batched = (self.batch_size > 1)
            
            if t_src is not None:
                t0_ch = time.perf_counter()
                if is_batched:
                    self._maps['tArr'][:, h_start:h_end] = t_src[:, :n_steps]
                else:
                    self._maps['tArr'][h_start:h_end] = t_src[:n_steps]
                ch_breakdown['tArr'] = time.perf_counter() - t0_ch
                
            if r_time and st_src is not None:
                t0_ch = time.perf_counter()
                if is_batched:
                    self._maps['stepTime'][:, h_start:h_end] = st_src[:, :n_steps]
                else:
                    self._maps['stepTime'][h_start:h_end] = st_src[:n_steps]
                ch_breakdown['stepTime'] = time.perf_counter() - t0_ch
                
            if (r_hist & 1) and v_src is not None:
                t0_ch = time.perf_counter()
                if is_batched:
                    self._maps['vHist'][:, :, h_start:h_end] = v_src[:, :, :n_steps]
                else:
                    self._maps['vHist'][:, h_start:h_end] = v_src[:, :n_steps]
                ch_breakdown['vHist'] = time.perf_counter() - t0_ch
                
            if self.num_devices > 0:
                if (r_hist & 2) and i_src is not None:
                    t0_ch = time.perf_counter()
                    if is_batched:
                        self._maps['iHist'][:, :, h_start:h_end] = i_src[:, :, :n_steps]
                    else:
                        self._maps['iHist'][:, h_start:h_end] = i_src[:, :n_steps]
                    ch_breakdown['iHist'] = time.perf_counter() - t0_ch
                    
                if (r_hist & 4) and s_src is not None:
                    t0_ch = time.perf_counter()
                    if is_batched:
                        self._maps['stateHist'][:, :, h_start:h_end] = s_src[:, :, :n_steps]
                    else:
                        self._maps['stateHist'][:, h_start:h_end] = s_src[:, :n_steps]
                    ch_breakdown['stateHist'] = time.perf_counter() - t0_ch
                    
                if (r_hist & 8) and f_src is not None:
                    t0_ch = time.perf_counter()
                    if is_batched:
                        self._maps['f22Hist'][:, :, h_start:h_end] = f_src[:, :, :n_steps]
                    else:
                        self._maps['f22Hist'][:, h_start:h_end] = f_src[:, :n_steps]
                    ch_breakdown['f22Hist'] = time.perf_counter() - t0_ch
                    
                if (r_hist & 16) and g_src is not None:
                    t0_ch = time.perf_counter()
                    if is_batched:
                        self._maps['gHist'][:, :, h_start:h_end] = g_src[:, :, :n_steps]
                    else:
                        self._maps['gHist'][:, h_start:h_end] = g_src[:, :n_steps]
                    ch_breakdown['gHist'] = time.perf_counter() - t0_ch
                    
                if (r_hist & 32) and temp_src is not None:
                    t0_ch = time.perf_counter()
                    if is_batched:
                        self._maps['tempHist'][:, :, h_start:h_end] = temp_src[:, :, :n_steps]
                    else:
                        self._maps['tempHist'][:, h_start:h_end] = temp_src[:, :n_steps]
                    ch_breakdown['tempHist'] = time.perf_counter() - t0_ch
                    
            tot_dur = time.perf_counter() - t_start_work
            SSDPipelineTelemetry.get_instance().record_chunk(n_steps, n_bytes, disp_lat, tot_dur, ch_breakdown)
                
        pool = ThreadPoolPingPongManager.get_instance()._writer_pool
        future = pool.submit(
            _do_write, host_start, host_end, completed_steps, bitmask, recordStepTiming,
            tArr, vHist, iHist, stateHist, f22Hist, gHist, tempHist, stepTimeHist, submit_time, chunk_bytes
        )
        self._writer_futures.append(future)
        if len(self._writer_futures) > 16:
            self._writer_futures = [f for f in self._writer_futures if not f.done()]
        return future

    def wait_for_writes(self):
        """
        Synchronously waits for all pending background chunk writes to complete.
        """
        if getattr(self, '_writer_futures', None):
            t0_wait = time.perf_counter()
            concurrent.futures.wait(self._writer_futures)
            self._writer_futures.clear()
            SSDPipelineTelemetry.get_instance().record_flush_wait(time.perf_counter() - t0_wait)

    def flush(self):
        self.wait_for_writes()
        for m in self._maps.values():
            if isinstance(m, np.memmap):
                try:
                    m.flush()
                except Exception:
                    pass

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        ThreadPoolPingPongManager.get_instance().unregister_instance(1)
        self.wait_for_writes()
        
        # Release shared file descriptor locks
        for fd in list(getattr(self, '_lock_fds', {}).values()):
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except Exception:
                pass
        if hasattr(self, '_lock_fds'):
            self._lock_fds.clear()

        for name in list(self._maps.keys()):
            m = self._maps.get(name)
            if m is not None and isinstance(m, np.memmap):
                try:
                    m.flush()
                except Exception:
                    pass
                try:
                    if hasattr(m, '_mmap') and m._mmap is not None:
                        m._mmap.close()
                except Exception:
                    pass
            self._maps[name] = None
        self._maps.clear()
        
        # Force garbage collection to ensure OS handles to mmaps are released before unlinking
        import gc
        gc.collect()

        for name, filepath in list(self._files.items()):
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except PermissionError:
                    try:
                        os.chmod(filepath, 0o777)
                        os.remove(filepath)
                    except Exception:
                        pass
                except Exception:
                    pass
        self._files.clear()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

class LazyHistoryDict(dict):
    def __init__(self, full_array, keys_ordered, active_len):
        super().__init__()
        self._full_array = full_array
        self._keys_ordered = keys_ordered if isinstance(keys_ordered, list) else list(keys_ordered)
        self._num_keys = len(self._keys_ordered)
        self._key_to_idx = {k: i for i, k in enumerate(self._keys_ordered)}
        self._active_len = active_len

    def __getitem__(self, key):
        return self._full_array[self._key_to_idx[key], :self._active_len]

    def __contains__(self, key):
        return key in self._key_to_idx

    def __len__(self):
        return self._num_keys

    def __iter__(self):
        return iter(self._keys_ordered)

    def keys(self):
        return self._keys_ordered

    def values(self):
        return list(self._full_array[:self._num_keys, :self._active_len])

    def items(self):
        active_slice = self._full_array[:self._num_keys, :self._active_len]
        return list(zip(self._keys_ordered, active_slice))

    def get(self, key, default=None):
        idx = self._key_to_idx.get(key)
        return self._full_array[idx, :self._active_len] if idx is not None else default

    def copy(self):
        active_slice = self._full_array[:self._num_keys, :self._active_len]
        return dict(zip(self._keys_ordered, active_slice))

    def __setitem__(self, key, value):
        raise TypeError("LazyHistoryDict is read-only")

    def __delitem__(self, key):
        raise TypeError("LazyHistoryDict is read-only")

    def clear(self):
        raise TypeError("LazyHistoryDict is read-only")

    def pop(self, key, default=None):
        raise TypeError("LazyHistoryDict is read-only")

    def popitem(self):
        raise TypeError("LazyHistoryDict is read-only")

    def setdefault(self, key, default=None):
        raise TypeError("LazyHistoryDict is read-only")

    def update(self, *args, **kwargs):
        raise TypeError("LazyHistoryDict is read-only")

    def __repr__(self):
        return f"<LazyHistoryDict with {self._num_keys} keys>"

    def __str__(self):
        return self.__repr__()

class LazyDeviceHistoryDict(dict):
    def __init__(self, full_array, num_devices, active_len):
        super().__init__()
        self._full_array = full_array
        self._num_devices = num_devices
        self._active_len = active_len

    def __getitem__(self, key):
        if type(key) is int:
            if 0 <= key < self._num_devices:
                return self._full_array[key, :self._active_len]
            raise KeyError(key)
        try:
            k = int(key)
            if 0 <= k < self._num_devices:
                return self._full_array[k, :self._active_len]
        except (ValueError, TypeError):
            pass
        raise KeyError(key)

    def __contains__(self, key):
        if type(key) is int:
            return 0 <= key < self._num_devices
        try:
            return 0 <= int(key) < self._num_devices
        except (ValueError, TypeError):
            return False

    def __len__(self):
        return self._num_devices

    def __iter__(self):
        return iter(range(self._num_devices))

    def keys(self):
        return range(self._num_devices)

    def values(self):
        return list(self._full_array[:self._num_devices, :self._active_len])

    def items(self):
        return list(enumerate(self._full_array[:self._num_devices, :self._active_len]))

    def get(self, key, default=None):
        if type(key) is int:
            return self._full_array[key, :self._active_len] if 0 <= key < self._num_devices else default
        try:
            k = int(key)
            return self._full_array[k, :self._active_len] if 0 <= k < self._num_devices else default
        except (ValueError, TypeError):
            pass
        return default

    def copy(self):
        return dict(enumerate(self._full_array[:self._num_devices, :self._active_len]))

    def __setitem__(self, key, value):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def __delitem__(self, key):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def clear(self):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def pop(self, key, default=None):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def popitem(self):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def setdefault(self, key, default=None):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def update(self, *args, **kwargs):
        raise TypeError("LazyDeviceHistoryDict is read-only")

    def __repr__(self):
        return f"<LazyDeviceHistoryDict with {self._num_devices} devices>"

    def __str__(self):
        return self.__repr__()

class SimulatorHistoryMixin:
    def _ensure_history_synced(self):
        store = getattr(self, '_disk_history_store', None)
        if store is not None and hasattr(store, 'wait_for_writes'):
            store.wait_for_writes()

    def getTime(self):
        """Returns the time array for the simulation."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        return t_arr

    def getVoltage(self, node=None, row=None, col=None):
        """Returns the voltage history array for a node, row, or col."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        targetNode = self._resolveNode(node, row, col)
        v_hist = self.vHistory.get(targetNode)
        if v_hist is None:
            raise ValueError(f"Target node {targetNode} not found in simulation.")
        return v_hist

    def getCurrent(self, topNode=None, bottomNode=None, row=None, col=None):
        """Returns the current history for a device via node connections or row/col."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        idx = self._resolveDeviceIndex(topNode, bottomNode, row, col)
        return self.iHistory[idx]
        
    def getState(self, topNode=None, bottomNode=None, row=None, col=None):
        """Returns the state 'n' history for a device via node connections or row/col."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        idx = self._resolveDeviceIndex(topNode, bottomNode, row, col)
        return self.stateHistory[idx]
        
    def getF22(self, topNode=None, bottomNode=None, row=None, col=None):
        """Returns the normalized state history (f22) for a device via node connections or row/col."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        idx = self._resolveDeviceIndex(topNode, bottomNode, row, col)
        return self.f22History[idx]

    def getConductance(self, topNode=None, bottomNode=None, row=None, col=None):
        """Returns the read conductance history in Siemens for a device, corrected for the read voltage."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        idx = self._resolveDeviceIndex(topNode, bottomNode, row, col)
        g_small = self.gHistory[idx]
        
        v_read = getattr(self, 'vRead', 0.1)
        if abs(v_read) < 1e-12:
            return g_small
            
        device_entry = self.devices[idx]
        dev_obj = device_entry['dev']
        
        # Reconstruct V_scale at each time step using stateHistory (n) and f22History
        if (not hasattr(self, 'stateHistory') or not self.stateHistory or idx not in self.stateHistory or
            not hasattr(self, 'f22History') or not self.f22History or idx not in self.f22History):
            return g_small

        n_hist = self.stateHistory[idx]
        f22_hist = self.f22History[idx]
        y_v = dev_obj._y_v if hasattr(dev_obj, '_y_v') and dev_obj._y_v is not None else dev_obj.lookups["vScale"]
        dev_gamma = dev_obj.gamma
        dev_nmax = float(dev_obj.nMax)
        
        # Direct O(1) vectorized linear interpolation over uniform lookup grid
        len_y = len(y_v)
        if len_y > 1 and y_v[0] == y_v[-1] and dev_gamma == 0.0:
            # Uniform constant V_scale fast-path (bypasses intermediate array allocations)
            inv_y0 = 1.0 / float(y_v[0]) if float(y_v[0]) != 0.0 else 0.0
            ratio = v_read * inv_y0
            ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
            if abs(v_read) > 1e-5:
                correction = math.sinh(ratio_clipped) / ratio
            else:
                correction = (math.sinh(ratio_clipped) / (ratio + 1e-30)) if abs(ratio) > 1e-5 else (1.0 + (ratio * ratio) * 0.16666666666666666)
            return g_small if correction == 1.0 else (g_small * correction)
            
        if len_y > 1:
            inv_nmax = 1.0 / (dev_nmax if dev_nmax > 1e-9 else 1e-9) if dev_nmax > 0.0 else 1.0
            n_scale = (len_y - 1.0) * inv_nmax
            n_scaled = np.clip(n_hist * n_scale, 0.0, len_y - 1.000001)
            idx_low = n_scaled.astype(np.int32, copy=False)
            frac = n_scaled - idx_low
            v0 = y_v[idx_low]
            vScale_interp = v0 + frac * (y_v[idx_low + 1] - v0)
        elif len_y == 1:
            vScale_interp = np.full_like(n_hist, y_v[0])
        else:
            vScale_interp = np.ones_like(n_hist)
        if dev_gamma != 0.0:
            denom = 1.0 + dev_gamma * f22_hist
            np.maximum(denom, 1e-15, out=denom)
            np.reciprocal(denom, out=denom)
            vScale_eff = vScale_interp * denom
        else:
            vScale_eff = vScale_interp
        
        ratio = v_read / vScale_eff
        ratio_clipped = np.clip(ratio, -50.0, 50.0)
        
        # Numerically stable sinh(ratio)/ratio using Taylor expansion for small values
        if abs(v_read) > 1e-5:
            inv_vread = 1.0 / v_read
            correction = np.sinh(ratio_clipped) * (vScale_eff * inv_vread)
        else:
            abs_ratio = np.abs(ratio)
            correction = np.where(
                abs_ratio > 1e-5,
                np.sinh(ratio_clipped) / (ratio + 1e-30),
                1.0 + (ratio * ratio) * 0.16666666666666666
            )
        
        return g_small * correction


    def getTemperature(self, topNode=None, bottomNode=None, row=None, col=None):
        """Returns the temperature history in Kelvin for a device."""
        self._ensure_history_synced()
        t_arr = self.tArr
        if t_arr is None:
            raise ValueError("Simulation has not been run yet.")
        idx = self._resolveDeviceIndex(topNode, bottomNode, row, col)
        if not hasattr(self, 'tempHistory') or not self.tempHistory or idx not in self.tempHistory:
            return np.full_like(t_arr, 298.15)
        return self.tempHistory[idx]
