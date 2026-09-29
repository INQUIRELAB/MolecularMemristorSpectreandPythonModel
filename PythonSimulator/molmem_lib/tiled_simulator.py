import numpy as np
import os
import sys
import time
import atexit
import psutil
import concurrent.futures
from .simulator import MolmemSimulator
from .sys_utils import DummyStream, handle_oom

_SYS_CPU_COUNT = os.cpu_count() or 1
_SYS_80_PERCENT_CORES = max(1, int(_SYS_CPU_COUNT * 0.8))

class MolmemTiledSimulator:
    """
    A macro-architecture simulator that wraps multiple identical MolmemSimulator tiles
    into a single large logical crossbar matrix. Automatically manages OpenMP/Python
    thread scheduling and orchestrates inference and weight updates across the tiles.
    """
    def __init__(self, instanceName="TiledArray", device='auto', backend='auto', vram_fraction=0.8):
        self.instanceName = instanceName
        self.device = device
        self.backend = backend
        self.vram_fraction = vram_fraction
        self.total_rows = 0
        self.total_cols = 0
        self.tile_rows = 0
        self.tile_cols = 0
        self.num_row_tiles = 0
        self.num_col_tiles = 0
        self.tiles = [] # 2D array [row_idx][col_idx]
        self.executor = None
        self.outer_workers = 1

    def _atexit_cleanup(self):
        try:
            if hasattr(self, '_registered_sim_count') and self._registered_sim_count > 0:
                from .history import ThreadPoolPingPongManager
                ThreadPoolPingPongManager.get_instance().unregister_instance(self._registered_sim_count)
                self._registered_sim_count = 0
        except Exception:
            pass
        try:
            if hasattr(self, 'executor') and self.executor is not None:
                self.executor.shutdown(wait=False)
        except Exception:
            pass

    def __del__(self):
        try:
            if hasattr(self, '_atexit_cleanup_ref'):
                atexit.unregister(self._atexit_cleanup_ref)
        except Exception:
            pass
        if self.executor is not None:
            self.executor.shutdown(wait=False)

    def addTiledCrossbarMatrix(self, total_rows, total_cols, tile_rows, tile_cols, deviceClass=None, device_type="default", weightBits=None, inputBits=None, outputBits=None, vWrite=0.9, vErase=-0.75, max_vWrite=5.0, max_pw=320e-9, detailedPrint=True, architecture='1T1R', UpdatesPer="Column", force_device_init=True, write_noise_intensity=0.0, read_noise_intensity=0.0, d2d_variation_intensity=0.0):
        kwargs = {
            'deviceClass': deviceClass, 'device_type': device_type, 'weightBits': weightBits, 'inputBits': inputBits, 'outputBits': outputBits,
            'vWrite': vWrite, 'vErase': vErase, 'max_vWrite': max_vWrite, 'max_pw': max_pw,
            'detailedPrint': detailedPrint, 'architecture': architecture,
            'UpdatesPer': UpdatesPer, 'force_device_init': force_device_init,
            'write_noise_intensity': write_noise_intensity, 'read_noise_intensity': read_noise_intensity, 'd2d_variation_intensity': d2d_variation_intensity
        }

        if total_rows % tile_rows != 0 or total_cols % tile_cols != 0:
            raise ValueError(f"Total matrix dimensions ({total_rows}x{total_cols}) must be evenly divisible by the tile dimensions ({tile_rows}x{tile_cols}).")

        self.total_rows = total_rows
        self.total_cols = total_cols
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols

        self.num_row_tiles = total_rows // tile_rows
        self.num_col_tiles = total_cols // tile_cols
        
        # Balance overall VRAM budget equally across all tiles
        total_tiles = self.num_row_tiles * self.num_col_tiles
        tile_vram_fraction = self.vram_fraction / total_tiles if total_tiles > 0 else self.vram_fraction
        
        # Register instances with ThreadPoolPingPongManager to guarantee dedicated ping-pong buffers per parallel tile
        from .history import ThreadPoolPingPongManager
        ThreadPoolPingPongManager.get_instance().register_instance(total_tiles)
        self._registered_sim_count = total_tiles
        
        # Instantiate the first tile to allow it to hardware profile
        first_tile = MolmemSimulator(instanceName=f"{self.instanceName}_0_0", device=self.device, backend=self.backend, vram_fraction=tile_vram_fraction)
        first_tile._worker_id = 0
        first_tile.addCrossbarMatrix(rows=tile_rows, cols=tile_cols, **kwargs)
        
        # Synchronously lock the hardware calibration phase to the first array instance.
        # This prevents tens of concurrent threads from each spawning their own
        # ProcessPoolExecutor during the 3D physics pulse-mapping phase.
        first_prog = getattr(first_tile, 'programmer', None)
        if first_prog is not None:
            if first_prog.stateMapPot is None or first_prog.stateMapDep is None or first_prog.gMapG is None:
                if detailedPrint:
                    print(f"\n[+] TiledSimulator: Initializing synchronous hardware calibration for {self.instanceName} limits...")
                first_prog.calibrate()
        
        # Synchronously profile parallelization on the main thread for the first tile.
        # This determines optimal core allocation and ensures subsequent tiles reuse the memory cache.
        if hasattr(first_tile, 'profileParallelization'):
            orig_dp = getattr(first_tile, 'detailedPrint', True)
            first_tile.detailedPrint = detailedPrint
            try:
                first_tile.profileParallelization()
            finally:
                first_tile.detailedPrint = orig_dp

        # Resolve hardware properties
        self.device = getattr(first_tile, 'device', self.device)
        self.backend = getattr(first_tile, 'backend', self.backend)
        optimal_cores = getattr(first_tile, 'optimal_cores', 1)
        target_cores = getattr(first_tile, 'target_cores', _SYS_80_PERCENT_CORES)
        self.is_cpu_backend = (getattr(first_tile, 'device', 'cpu') == 'cpu' or getattr(first_tile, 'backend', 'numba') in ['numba', 'numpy', 'cpu', None] or self.backend in ['numba', 'numpy', 'cpu', None])
        
        # On GPU, allow up to all tiles to submit concurrently
        if not self.is_cpu_backend:
            self.outer_workers = max(1, min(32, total_tiles))
        else:
            self.outer_workers = max(1, target_cores // optimal_cores)
            
        self.tiles = [[None for _ in range(self.num_col_tiles)] for _ in range(self.num_row_tiles)]
        self.tiles[0][0] = first_tile
        
        if detailedPrint:
            print(f"\n[+] TiledSimulator Profiling [Backend: {self.backend} (Device: {self.device})]: Creating {self.num_row_tiles * self.num_col_tiles} tiles of {tile_rows}x{tile_cols}.")
            print(f"    - Inner Numba threads per tile: {optimal_cores}")
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Initialized ThreadPoolExecutor: TiledSimulator | Workers={self.outer_workers} | Platform={'CPU' if self.is_cpu_backend else 'GPU'}")
        except Exception:
            pass
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.outer_workers)
        
        # Register simulator instance for Ctrl+C cleanup
        try:
            import molmem_lib
            molmem_lib.register_simulator(self)
        except Exception:
            pass
        
        # Register atexit cleanup to prevent thread deadlocks on exit
        self._atexit_cleanup_ref = self._atexit_cleanup
        atexit.register(self._atexit_cleanup_ref)

        # Instantiate remaining tiles and assign dedicated worker IDs
        for r in range(self.num_row_tiles):
            for c in range(self.num_col_tiles):
                tile_idx = r * self.num_col_tiles + c
                if r == 0 and c == 0:
                    first_tile._worker_id = tile_idx
                    continue # First tile already built
                tile = MolmemSimulator(instanceName=f"{self.instanceName}_{r}_{c}", device=self.device, backend=self.backend, vram_fraction=tile_vram_fraction)
                tile._worker_id = tile_idx
                tile.addCrossbarMatrix(rows=tile_rows, cols=tile_cols, **kwargs)
                tile.optimal_cores = optimal_cores
                tile.use_parallel_optimized = getattr(first_tile, 'use_parallel_optimized', False)
                if first_prog is not None:
                    tile_prog = getattr(tile, 'programmer', None)
                    if tile_prog is not None:
                        tile_prog.calibrate()
                self.tiles[r][c] = tile

        self.inQ = getattr(first_tile, 'inQ', None)
        self.adc = getattr(first_tile, 'adc', None)
        self.programmer = getattr(first_tile, 'programmer', None)
        self.recordHistory = False

        # Dedicated hardware stream per tile on GPU
        self.tile_streams = None
        if not self.is_cpu_backend:
            try:
                import torch
                if torch.cuda.is_available():
                    dev_obj = torch.device('cuda' if (self.device == 'auto' or str(self.device).startswith('cuda')) else self.device)
                    self.tile_streams = [
                        [torch.cuda.Stream(device=dev_obj) for _ in range(self.num_col_tiles)]
                        for _ in range(self.num_row_tiles)
                    ]
            except Exception:
                self.tile_streams = None

        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[GPU_TILED_GRID] Tiled grid created | Grid={self.num_row_tiles}x{self.num_col_tiles} tiles ({tile_rows}x{tile_cols} each) | TotalDevs={self.total_rows*self.total_cols} | Platform={'CPU' if self.is_cpu_backend else 'GPU'}")
        except Exception:
            pass

    @handle_oom
    def updateTiledCrossbarWeights(self, targetWeightsNormalized, detailedPrint=True, prefix="", fineTune=True, recordHistory=False, silentUpdate=False, **kwargs):
        if silentUpdate:
            detailedPrint = False

        if not isinstance(targetWeightsNormalized, np.ndarray):
            targetWeightsNormalized = np.asarray(targetWeightsNormalized)

        if targetWeightsNormalized.shape != (self.total_rows, self.total_cols):
            raise ValueError(f"Weight matrix shape mismatch. Expected ({self.total_rows}, {self.total_cols}), got {targetWeightsNormalized.shape}")

        if not np.all(np.isfinite(targetWeightsNormalized)):
            raise ValueError("Target weight matrix contains NaN or Infinite values.")
        
        t_rows = self.tile_rows
        t_cols = self.tile_cols
        inst_name = self.instanceName
        tiles = self.tiles
        n_row_tiles = self.num_row_tiles
        n_col_tiles = self.num_col_tiles
        total_tiles = n_row_tiles * n_col_tiles
        is_cpu = self.is_cpu_backend
        backend_str = getattr(self, 'backend', 'hybrid')
        
        # Multi-Tenant resource budgeting across all tiles
        total_cpu_budget = max(1, int((os.cpu_count() or 32) * 0.8))
        per_tile_cpu_budget = max(1, total_cpu_budget // total_tiles)
        
        cu_count = 32
        threads_per_cu = 1024
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                cu_count = getattr(props, 'multi_processor_count', 32) or 32
                threads_per_cu = getattr(props, 'max_threads_per_multi_processor', 1024) or 1024
        except Exception:
            pass
            
        per_tile_cu_budget = max(1, cu_count // total_tiles)
        per_tile_max_gpu_devs = int(per_tile_cu_budget * threads_per_cu)
        
        for r in range(n_row_tiles):
            for c in range(n_col_tiles):
                tile = tiles[r][c]
                opt = getattr(tile, 'optimal_cores', per_tile_cpu_budget) or per_tile_cpu_budget
                tile._thread_budget = max(1, min(int(opt), per_tile_cpu_budget))
                tile._cu_budget = per_tile_cu_budget
                tile._max_gpu_device_batch = per_tile_max_gpu_devs

        def _update_single_tile(r, c, arg_tile, arg_weights, arg_prefix):
            threading_layer = "unknown"
            try:
                from .sys_utils import get_numba_threading_layer
                threading_layer = get_numba_threading_layer()
            except Exception:
                pass

            is_hybrid = getattr(arg_tile, 'backend', 'hybrid') in ['hybrid', 'auto']
            needs_lock = (threading_layer != "tbb") and (is_cpu or is_hybrid)

            if self.tile_streams is not None and not is_cpu:
                import torch
                stream = self.tile_streams[r][c]
                arg_tile._cuda_stream = stream
                if needs_lock:
                    from .simulator import _GLOBAL_CPU_CROSSBAR_LOCK
                    with _GLOBAL_CPU_CROSSBAR_LOCK:
                        with torch.cuda.stream(stream):
                            arg_tile.updateCrossbarWeights(
                                targetWeightsNormalized=arg_weights, 
                                detailedPrint=detailedPrint, 
                                prefix=arg_prefix, 
                                fineTune=fineTune, 
                                recordHistory=recordHistory,
                                silentUpdate=not detailedPrint
                            )
                else:
                    with torch.cuda.stream(stream):
                        arg_tile.updateCrossbarWeights(
                            targetWeightsNormalized=arg_weights, 
                            detailedPrint=detailedPrint, 
                            prefix=arg_prefix, 
                            fineTune=fineTune, 
                            recordHistory=recordHistory,
                            silentUpdate=not detailedPrint
                        )
                return 1

            if needs_lock:
                from .simulator import _GLOBAL_CPU_CROSSBAR_LOCK
                with _GLOBAL_CPU_CROSSBAR_LOCK:
                    arg_tile.updateCrossbarWeights(
                        targetWeightsNormalized=arg_weights, 
                        detailedPrint=detailedPrint, 
                        prefix=arg_prefix, 
                        fineTune=fineTune, 
                        recordHistory=recordHistory,
                        silentUpdate=not detailedPrint
                    )
            else:
                arg_tile.updateCrossbarWeights(
                    targetWeightsNormalized=arg_weights, 
                    detailedPrint=detailedPrint, 
                    prefix=arg_prefix, 
                    fineTune=fineTune, 
                    recordHistory=recordHistory,
                    silentUpdate=not detailedPrint
                )
            return 1
            
        old_stdout = sys.__stdout__
        if old_stdout is None:
            old_stdout = DummyStream()
        prog_start = time.perf_counter()
        completed = 0
        inv_total_tiles = 1.0 / max(1, total_tiles)

        futures = [
            self.executor.submit(
                _update_single_tile,
                r, c,
                tiles[r][c],
                targetWeightsNormalized[r * t_rows : (r + 1) * t_rows, c * t_cols : (c + 1) * t_cols],
                f"[Tile {r},{c}] [Backend: {backend_str}] {prefix}" if prefix else f"[{inst_name}_{r}_{c}]"
            )
            for r in range(n_row_tiles)
            for c in range(n_col_tiles)
        ]
                
        from .sys_utils import isMolmemLibSilenced
        is_silenced = isMolmemLibSilenced()

        if not detailedPrint and not is_silenced:
            pfx = f"[{self.instanceName}] [Backend: {backend_str}] " if not prefix else f"  {prefix} "
            old_stdout.write(f"\r  {pfx}[{' '*20}]   0% | Submitting {total_tiles} parallel tiles... ".ljust(95))
            old_stdout.flush()
        
        for future in concurrent.futures.as_completed(futures):
            items_done = future.result() # Re-raise internal exceptions immediately
            completed += items_done
            
            if not detailedPrint and not is_silenced:
                prog = min(100.0, (completed * inv_total_tiles) * 100.0)
                bar = '#' * int(prog * 0.2)
                pfx = f"[{self.instanceName}] [Backend: {backend_str}] " if not prefix else f"  {prefix} "
                old_stdout.write(f"\r  {pfx}[{bar.ljust(20)}] {prog:3.0f}% | Programmed {completed}/{total_tiles} Tiles ".ljust(95))
                old_stdout.flush()

        if self.tile_streams is not None and not is_cpu:
            for r in range(n_row_tiles):
                for c in range(n_col_tiles):
                    try:
                        self.tile_streams[r][c].synchronize()
                    except Exception:
                        pass
                
        if not detailedPrint:
            elapsed = time.perf_counter() - prog_start
            pfx = f"[{self.instanceName}] [Backend: {backend_str}] " if not prefix else f"  {prefix} "
            
            if prefix:
                old_stdout.write(f"\r  {pfx}[{'#'*20}] 100% | Programmed in {elapsed:.3f}s ".ljust(95))
                old_stdout.flush()
            else:
                wStr = np.array2string(targetWeightsNormalized, precision=2, separator=', ', suppress_small=True)
                if len(wStr) > 40:
                    wStr = wStr[:37] + "..."
                
                old_stdout.write(f"\r  {pfx}[{'#'*20}] 100% | HW: {wStr} | {elapsed:.3f}s".ljust(95))
                old_stdout.write("\n\n")
                old_stdout.flush()

    def updateTileCrossbarWeights(self, tile_r, tile_c, targetWeightsNormalized, detailedPrint=True, prefix="", fineTune=True, recordHistory=False, silentUpdate=False, **kwargs):
        if silentUpdate:
            detailedPrint = False

        if not (0 <= tile_r < self.num_row_tiles) or not (0 <= tile_c < self.num_col_tiles):
            raise ValueError(f"Tile coordinates ({tile_r}, {tile_c}) out of bounds. Matrix has {self.num_row_tiles}x{self.num_col_tiles} tiles.")
            
        if targetWeightsNormalized.shape != (self.tile_rows, self.tile_cols):
            raise ValueError(f"Weight matrix shape mismatch. Expected ({self.tile_rows}, {self.tile_cols}), got {targetWeightsNormalized.shape}")
            

        tile = self.tiles[tile_r][tile_c]
        backend_str = getattr(self, 'backend', 'hybrid')
        tile_prefix = f"[Tile {tile_r},{tile_c}] [Backend: {backend_str}] {prefix}" if prefix else f"[{self.instanceName}_{tile_r}_{tile_c}] [Backend: {backend_str}]"
        if self.tile_streams is not None and not self.is_cpu_backend:
            import torch
            stream = self.tile_streams[tile_r][tile_c]
            tile._cuda_stream = stream
            with torch.cuda.stream(stream):
                tile.updateCrossbarWeights(
                    targetWeightsNormalized=targetWeightsNormalized, 
                    detailedPrint=detailedPrint, 
                    prefix=tile_prefix, 
                    fineTune=fineTune, 
                    recordHistory=recordHistory,
                    silentUpdate=silentUpdate or (not detailedPrint)
                )
                stream.synchronize()
                return
        tile.updateCrossbarWeights(
            targetWeightsNormalized=targetWeightsNormalized, 
            detailedPrint=detailedPrint, 
            prefix=tile_prefix, 
            fineTune=fineTune, 
            recordHistory=recordHistory,
            silentUpdate=silentUpdate or (not detailedPrint)
        )

    @handle_oom
    def passTiledCrossbarInput(self, xNormalized, detailedPrint=True, appendHistory=False, stream=None, return_torch=False, **kwargs):
        """
        Passes a normalized Input Vector (1D) or a 2D batch of input vectors through the tiled crossbar array.

        Args:
            xNormalized: 1D input vector [total_rows] or 2D batch [batch_size, total_rows] (NumPy array or PyTorch Tensor).
            detailedPrint: Whether to print verbose progress.
            appendHistory: Whether to append to transient waveform history (supported only for 1D vectors).
            stream: Optional PyTorch CUDA/HIP Stream to execute on.
            return_torch: If True, returns output as a PyTorch Tensor.
        """
        try:
            import torch
            is_torch_input = isinstance(xNormalized, torch.Tensor)
        except ImportError:
            is_torch_input = False

        if is_torch_input:
            x_np = xNormalized.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(xNormalized, dtype=np.float32)

        flatten_output_1d = False
        if x_np.ndim == 1:
            if len(x_np) != self.total_rows:
                raise ValueError(f"Input vector length mismatch. Expected {self.total_rows}, got {len(x_np)}")
            if not appendHistory:
                # 1D Fastpath: Route single vector through the fused batch-parallel solver as (1, total_rows)
                # completely bypassing the heavy picosecond-level adaptive Runge-Kutta circuit ODE engine
                x_np = x_np.reshape(1, self.total_rows)
                is_batch = True
                batch_size = 1
                flatten_output_1d = True
            else:
                is_batch = False
        elif x_np.ndim == 2:
            batch_size, input_rows = x_np.shape
            if input_rows != self.total_rows:
                raise ValueError(f"Input vector length mismatch. Expected {self.total_rows}, got {input_rows}")
            if appendHistory:
                raise ValueError("History logging (appendHistory=True) is not supported when passing inputs in batch parallel mode.")
            is_batch = True
        else:
            raise ValueError(f"Expected 1D vector or 2D batch, got ndim={x_np.ndim}")

        if self.num_row_tiles == 1 and self.num_col_tiles == 1:
            return self.tiles[0][0].passInput(
                xNormalized, 
                detailedPrint=detailedPrint, 
                appendHistory=appendHistory,
                stream=stream,
                return_torch=return_torch
            )

        t_cols = self.tile_cols
        t_rows = self.tile_rows
        n_row_tiles = self.num_row_tiles
        n_col_tiles = self.num_col_tiles
        tiles = self.tiles

        # Vectorized input partitioning across row tiles via zero-copy C-level split
        x_row_chunks = np.split(x_np, n_row_tiles, axis=1 if is_batch else 0) if n_row_tiles > 1 else [x_np]

        if self.is_cpu_backend or self.outer_workers <= 1:
            if n_row_tiles == 1:
                y_cols = [
                    tiles[0][c].passInput(x_np, detailedPrint=detailedPrint, appendHistory=appendHistory, return_torch=False)
                    for c in range(n_col_tiles)
                ]
            else:
                y_grid = [
                    [tiles[r][c].passInput(x_row_chunks[r], detailedPrint=detailedPrint, appendHistory=appendHistory, return_torch=False)
                     for c in range(n_col_tiles)]
                    for r in range(n_row_tiles)
                ]
                y_cols = [
                    np.add.reduce([y_grid[r][c] for r in range(n_row_tiles)])
                    for c in range(n_col_tiles)
                ]
            y_out = np.concatenate(y_cols, axis=-1) if n_col_tiles > 1 else y_cols[0]
        else:
            def _pass_single(r, c, x_chunk):
                tile = self.tiles[r][c]
                stream_rc = self.tile_streams[r][c] if (self.tile_streams is not None and not self.is_cpu_backend) else stream
                y_tile = tile.passInput(x_chunk, detailedPrint=detailedPrint, appendHistory=appendHistory, stream=stream_rc, return_torch=False)
                return r, c, y_tile

            futures = [
                self.executor.submit(_pass_single, r, c, x_row_chunks[r])
                for r in range(n_row_tiles)
                for c in range(n_col_tiles)
            ]

            y_grid = [[None] * n_col_tiles for _ in range(n_row_tiles)]
            for future in concurrent.futures.as_completed(futures):
                r, c, y_tile = future.result()
                y_grid[r][c] = y_tile

            if n_row_tiles == 1:
                y_cols = y_grid[0]
            else:
                y_cols = [
                    np.add.reduce([y_grid[r][c] for r in range(n_row_tiles)])
                    for c in range(n_col_tiles)
                ]
            y_out = np.concatenate(y_cols, axis=-1) if n_col_tiles > 1 else y_cols[0]

        if self.tile_streams is not None and not self.is_cpu_backend:
            for r in range(n_row_tiles):
                for c in range(n_col_tiles):
                    try:
                        self.tile_streams[r][c].synchronize()
                    except Exception:
                        pass

        if flatten_output_1d:
            y_out = y_out[0]

        if return_torch:
            try:
                import torch
                if is_torch_input:
                    return torch.as_tensor(y_out, device=xNormalized.device, dtype=torch.float32)
                elif hasattr(self, 'device') and str(self.device).startswith('cuda'):
                    return torch.as_tensor(y_out, device=self.device, dtype=torch.float32)
                else:
                    return torch.from_numpy(y_out)
            except Exception:
                pass

        return y_out

    @handle_oom
    def readTiledCrossbarMatrix(self, detailedPrint=True, purpose=None, **kwargs):
        if self.num_row_tiles == 1 and self.num_col_tiles == 1:
            return self.tiles[0][0].readCrossbarMatrix(detailedPrint=detailedPrint, purpose=purpose)
            
        gMatrix = np.empty((self.total_rows, self.total_cols), dtype=np.float64)
        cMatrix = np.empty((self.total_rows, self.total_cols), dtype=np.int32)
        
        def _read_single(r, c):
            tile = self.tiles[r][c]
            tile_g, tile_c = tile.readCrossbarMatrix(detailedPrint=detailedPrint, purpose=purpose)
            return r, c, tile_g, tile_c

        t_cols = self.tile_cols
        t_rows = self.tile_rows
        n_row_tiles = self.num_row_tiles
        n_col_tiles = self.num_col_tiles
        tiles = self.tiles
        if n_row_tiles == 1:
            tile_row = tiles[0]
            futures = [
                (c, self.executor.submit(tile_row[c].readCrossbarMatrix, detailedPrint=detailedPrint, purpose=purpose))
                for c in range(n_col_tiles)
            ]
            for c, future in futures:
                tile_g, tile_c = future.result()
                c_start = c * t_cols
                c_end = c_start + t_cols
                gMatrix[:, c_start:c_end] = tile_g
                cMatrix[:, c_start:c_end] = tile_c
        else:
            futures = [
                self.executor.submit(_read_single, r, c)
                for r in range(n_row_tiles)
                for c in range(n_col_tiles)
            ]
            for future in concurrent.futures.as_completed(futures):
                r, c, tile_g, tile_c = future.result()
                r_start = r * t_rows
                r_end = r_start + t_rows
                c_start = c * t_cols
                c_end = c_start + t_cols
                gMatrix[r_start:r_end, c_start:c_end] = tile_g
                cMatrix[r_start:r_end, c_start:c_end] = tile_c
            
        return gMatrix, cMatrix

    def _conductanceToWeights(self, gMatrixRaw):
        first_tile = self.tiles[0][0]
        return first_tile._conductanceToWeights(gMatrixRaw)

    def saveStates(self):
        """Creates deep physical snapshots for all sub-array tiles in the tiled crossbar."""
        return [[self.tiles[r][c].saveStates() for c in range(self.num_col_tiles)] for r in range(self.num_row_tiles)]

    def restoreStates(self, states):
        """Restores physical internal states for all sub-array tiles in the tiled crossbar."""
        for r in range(self.num_row_tiles):
            for c in range(self.num_col_tiles):
                self.tiles[r][c].restoreStates(states[r][c])

    # API Compatibility Aliases with MolmemSimulator
    passInput = passTiledCrossbarInput
    updateCrossbarWeights = updateTiledCrossbarWeights
    readCrossbarMatrix = readTiledCrossbarMatrix
