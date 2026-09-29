import sys
import numpy as np
import math
import bisect

class BaseSource:
    def __init__(self, node):
        self.node = node

    def get_signature(self):
        return (self.__class__.__name__, self.node, id(self))

    def getVoltage(self, t):
        raise NotImplementedError

    def getNextEdge(self, t_current):
        return float('inf')

    def get_all_edges(self, tStart, tEnd):
        schedule = []
        t_search = tStart
        while t_search < tEnd:
            edge = self.getNextEdge(t_search)
            if edge == float('inf') or edge > tEnd + 1e-12:
                break
            schedule.append(edge)
            t_search = edge + 1e-18
        return schedule

_EMPTY_LIST = []
_EMPTY_ARRAY = np.empty(0, dtype=np.float64)

class DcSource(BaseSource):
    def __init__(self, node, voltage):
        super().__init__(node)
        self.voltage = voltage

    def get_signature(self):
        return ('dc', self.node, float(self.voltage))

    def getVoltage(self, t):
        if type(t) is float or type(t) is int:
            return self.voltage
        if isinstance(t, np.ndarray):
            return np.full_like(t, self.voltage)
        return self.voltage

    def get_all_edges(self, tStart, tEnd):
        return _EMPTY_ARRAY

class PulseSource(BaseSource):
    def __init__(self, node, amp, period, width, delay=0.0, max_pulses=None):
        super().__init__(node)
        self.amp = amp
        self.period = period
        self.width = width
        self.delay = delay
        self.max_pulses = max_pulses
        self.inv_period = 1.0 / period if period != 0.0 else 0.0
        self.t_cutoff = self.delay + self.max_pulses * self.period if self.max_pulses is not None else None
        self.effective_width = self.width - 1e-13
        self.effective_cutoff = (self.t_cutoff - 1e-15) if self.t_cutoff is not None else None

    def get_signature(self):
        return ('pulse', self.node, float(self.amp), float(self.period), float(self.width), float(self.delay), self.max_pulses)

    def getVoltage(self, t):
        if type(t) is float or type(t) is int:
            delay = self.delay
            if t < delay:
                return 0.0
            cutoff = self.t_cutoff
            if cutoff is not None and t >= cutoff:
                return 0.0
            tActive = t - delay
            cycles = tActive * self.inv_period
            cycle_idx = int(cycles + 1e-11)
            tCycle = tActive - cycle_idx * self.period
            return self.amp if tCycle < self.effective_width else 0.0
            
        delay = self.delay
        inv_p = self.inv_period
        period = self.period
        eff_w = self.effective_width
        amp = self.amp
        cutoff = self.t_cutoff
        if isinstance(t, np.ndarray):
            out = np.zeros_like(t)
            mask = t >= delay
            if cutoff is not None:
                mask = mask & (t < cutoff)
            if np.any(mask):
                tActive = t[mask] - delay
                cycles = tActive * inv_p
                cycle_idx = np.floor(cycles + 1e-11)
                tCycle = tActive - cycle_idx * period
                out[mask] = (tCycle < eff_w) * amp
            return out
            
        if t < delay:
            return 0.0
        if cutoff is not None and t >= cutoff:
            return 0.0
        tActive = t - delay
        cycles = tActive * inv_p
        cycle_idx = int(cycles + 1e-11)
        tCycle = tActive - cycle_idx * period
        return amp if tCycle < eff_w else 0.0

    def getNextEdge(self, t_current):
        delay = self.delay
        if t_current < delay:
            return delay
        eff_cutoff = self.effective_cutoff
        if eff_cutoff is not None and t_current >= eff_cutoff:
            return float('inf')
        tActive = t_current - delay
        
        inv_p = self.inv_period
        period = self.period
        width = self.width
        
        cycles = tActive * inv_p
        cycle_idx = int(cycles + 1e-11)
        
        cycle_start = delay + cycle_idx * period
        t_falling = cycle_start + width
        if t_falling > t_current + 1e-15:
            return t_falling
            
        t_next_rising = cycle_start + period
        if eff_cutoff is not None and t_next_rising >= eff_cutoff:
            return float('inf')
        return t_next_rising

    def get_all_edges(self, tStart, tEnd):
        delay = self.delay
        if tEnd < delay:
            return _EMPTY_LIST
        
        inv_p = self.inv_period
        period = self.period
        width = self.width
        max_p = self.max_pulses
        
        k_floor_start = math.floor((tStart - delay) * inv_p)
        k_min = 0 if k_floor_start < 0 else k_floor_start
        k_max = math.floor((tEnd - delay) * inv_p) + 1
        if max_p is not None:
            max_limit = max_p - 1
            k_max = k_max if k_max < max_limit else max_limit
            
        if k_min > k_max:
            return _EMPTY_LIST
        
        pre_dt = min(1e-10, width * 0.001) if width > 0 else 1e-11
        if k_min == k_max:
            rising = delay + k_min * period
            falling = rising + width
            return [e for e in (rising - pre_dt, rising, falling - pre_dt, falling) if tStart - 1e-15 <= e <= tEnd + 1e-15]

        k = np.arange(k_min, k_max + 1)
        rising = delay + k * period
        
        n_k = k_max - k_min + 1
        edges = np.empty(n_k << 2, dtype=np.float64)
        falling = rising + width
        edges[1::4] = rising
        edges[0::4] = rising - pre_dt
        edges[3::4] = falling
        edges[2::4] = falling - pre_dt
        
        i_start = np.searchsorted(edges, tStart - 1e-15, side='left')
        i_end = np.searchsorted(edges, tEnd + 1e-15, side='right')
        return edges[i_start:i_end]
        
class PwlSource(BaseSource):
    def __init__(self, node, t_points, v_points):
        super().__init__(node)
        self.t_points = np.ascontiguousarray(t_points, dtype=np.float64)
        self.v_points = np.ascontiguousarray(v_points, dtype=np.float64)

    def get_signature(self):
        return ('pwl', self.node, len(self.t_points), float(self.t_points[0]) if len(self.t_points) else 0.0, float(self.t_points[-1]) if len(self.t_points) else 0.0, float(self.v_points[0]) if len(self.v_points) else 0.0, float(self.v_points[-1]) if len(self.v_points) else 0.0)

    def getVoltage(self, t):
        if type(t) is float or type(t) is int:
            t_pts = self.t_points
            t_len = len(t_pts)
            if t_len == 0:
                return 0.0
            if t <= t_pts[0]:
                return self.v_points[0]
            if t >= t_pts[-1]:
                return self.v_points[-1]
            idx = bisect.bisect_right(t_pts, t) - 1
            idx_next = idx + 1
            t0 = t_pts[idx]
            t1 = t_pts[idx_next]
            dt = t1 - t0
            v0 = self.v_points[idx]
            v1 = self.v_points[idx_next]
            if dt < 1e-15:
                return v1
            return v0 + (t - t0) * (v1 - v0) / dt
            
        t_pts = self.t_points
        t_len = len(t_pts)
        if t_len == 0:
            return np.zeros_like(t)
        return np.interp(t, t_pts, self.v_points)

    def getNextEdge(self, t_current):
        t_pts = self.t_points
        t_len = len(t_pts)
        if t_len == 0:
            return float('inf')
        t_thresh = t_current + 1e-15
        if t_thresh >= t_pts[-1]:
            return float('inf')
        idx = bisect.bisect_right(t_pts, t_thresh)
        if idx < t_len:
            return t_pts[idx]
        return float('inf')

    def get_all_edges(self, tStart, tEnd):
        t_pts = self.t_points
        t_len = len(t_pts)
        if t_len == 0:
            return _EMPTY_ARRAY
        first_t = t_pts[0]
        last_t = t_pts[-1]
        if tEnd < first_t - 1e-15 or tStart > last_t + 1e-15:
            return _EMPTY_ARRAY
        if first_t >= tStart - 1e-15 and last_t <= tEnd + 1e-15:
            return t_pts
        i_start = np.searchsorted(t_pts, tStart - 1e-15, side='left')
        i_end = np.searchsorted(t_pts, tEnd + 1e-15, side='right')
        return t_pts[i_start:i_end]

class BSource(BaseSource):
    _FRACTIONS = (0.25, 0.5, 0.75)
    _warned_funcs = set()

    def __init__(self, node, vFunc, sources, edge_points=None, source_intervals=None):
        super().__init__(node)
        self.vFunc = vFunc
        self.sources = sources
        
        # Automatically schedule steps 1ns before each transition to capture pre-transition state (e.g. 0V)
        if edge_points:
            expanded = set(edge_points)
            expanded.update(p - 1e-9 for p in edge_points)
            self.edge_points = sorted(expanded)
        else:
            self.edge_points = []
        
        self.source_intervals = source_intervals
        self._active_intervals_cache = None
        self._segment_source_map = None
        self._cached_ufunc = None

    def get_signature(self):
        src_sigs = tuple(s.get_signature() if hasattr(s, 'get_signature') else (s.__class__.__name__, id(s)) for s in self.sources)
        return ('bsrc', self.node, getattr(self.vFunc, '__name__', id(self.vFunc)), len(self.sources), src_sigs)

    def get_active_intervals(self, tStart, tEnd):
        cache_key = (tStart, tEnd)
        if self._active_intervals_cache is not None and self._active_intervals_cache[0] == cache_key:
            return self._active_intervals_cache[1]
            
        if not self.edge_points:
            self._segment_source_map = {(tStart, tEnd): None}
            self._active_intervals_cache = (cache_key, [(tStart, tEnd)])
            return [(tStart, tEnd)]
            
        # Extract transition boundaries within simulation range using bisect
        edge_pts = self.edge_points
        i_left = bisect.bisect_right(edge_pts, tStart)
        i_right = bisect.bisect_left(edge_pts, tEnd)
        boundaries = [tStart, *edge_pts[i_left:i_right], tEnd]
        
        active = []
        segment_source_map = {}
        n_segs = len(boundaries) - 1
        fracs = self._FRACTIONS
        srcs = self.sources
        v_func = self.vFunc
        pulse_srcs = [
            (src.period, src.width, src.delay, src.inv_period)
            for src in srcs
            if isinstance(src, PulseSource) and src.period > 0.0 and src.width > 0.0
        ]
        for i in range(n_segs):
            t_start_val = boundaries[i]
            t_end_val = boundaries[i + 1]
            dt_seg = t_end_val - t_start_val
            key = (t_start_val, t_end_val)
            
            is_active = False
            matched_idx = None
            test_points = [t_start_val + f * dt_seg for f in fracs]
            for period, width, delay, inv_p in pulse_srcs:
                # Find the first rising edge at or after t_start_val
                k = math.ceil((t_start_val - delay) * inv_p)
                t_high = delay + k * period + 0.1 * width
                if t_start_val <= t_high < t_end_val:
                    test_points.append(t_high)
                            
            for t_test in test_points:
                try:
                    v_test = v_func(t_test, srcs)
                    if abs(v_test) > 1e-15:
                        is_active = True
                        # Check which source or scaled source matches v_test
                        for src_idx, src in enumerate(srcs):
                            try:
                                if isinstance(src, BaseSource):
                                    v_src = src.getVoltage(t_test)
                                else:
                                    v_src = float(src)
                                if abs(v_src - v_test) < 1e-12:
                                    matched_idx = (src_idx, 1.0)
                                    break
                                elif abs(v_src) > 1e-12:
                                    scale = v_test / v_src
                                    if abs(scale * v_src - v_test) < 1e-11:
                                        matched_idx = (src_idx, scale)
                                        break
                            except Exception:
                                pass
                        if matched_idx is None and is_active:
                            # Test if it's a constant DC voltage across this interval
                            matched_idx = ('const', float(v_test))
                        if matched_idx is not None:
                            break
                except Exception:
                    is_active = True
                    matched_idx = None
                    break
            
            segment_source_map[key] = matched_idx
            if is_active:
                active.append(key)
                
        self._active_intervals_cache = (cache_key, active)
        self._segment_source_map = segment_source_map
        return active

    def getVoltage(self, t):
        if isinstance(t, np.ndarray):
            if len(t) == 0:
                return np.zeros_like(t)
            try:
                res = self.vFunc(t, self.sources)
                if isinstance(res, np.ndarray) and res.shape == t.shape:
                    return res
            except Exception:
                pass
                
            active_intervals = self.get_active_intervals(float(t[0]), float(t[-1]))
            
            # Check if all active intervals mapped to a sub-source
            can_vectorize_piecewise = (self._segment_source_map is not None)
            if can_vectorize_piecewise:
                for interval in active_intervals:
                    if self._segment_source_map.get(interval) is None:
                        can_vectorize_piecewise = False
                        break
            
            if can_vectorize_piecewise:
                out = np.zeros_like(t)
                srcs = self.sources
                seg_map = self._segment_source_map
                t_len = len(t)
                t_last = float(t[-1]) if t_len > 0 else 0.0
                for interval in active_intervals:
                    t_start, t_end = interval
                    # Use binary search to find boundaries instead of creating boolean arrays
                    if t_len > 0 and abs(t_end - t_last) < 1e-11:
                        start_idx = np.searchsorted(t, t_start - 1e-15, side='left')
                        end_idx = np.searchsorted(t, t_end + 1e-15, side='right')
                    else:
                        start_idx = np.searchsorted(t, t_start - 1e-15, side='left')
                        end_idx = np.searchsorted(t, t_end - 1e-15, side='left')
                    
                    if start_idx < end_idx:
                        src_match = seg_map[interval]
                        if isinstance(src_match, tuple):
                            if src_match[0] == 'const':
                                out[start_idx:end_idx] = src_match[1]
                                continue
                            src_idx, scale = src_match
                        else:
                            src_idx, scale = src_match, 1.0
                        src = srcs[src_idx]
                        if isinstance(src, BaseSource):
                            if scale == 1.0:
                                out[start_idx:end_idx] = src.getVoltage(t[start_idx:end_idx])
                            elif scale == 0.0:
                                out[start_idx:end_idx] = 0.0
                            else:
                                out[start_idx:end_idx] = src.getVoltage(t[start_idx:end_idx]) * scale
                        else:
                            out[start_idx:end_idx] = float(src) * scale
                return out
                
            # Fallback to slow loop if vectorization not possible
            func_name = getattr(self.vFunc, '__name__', 'custom_function')
            if func_name not in BSource._warned_funcs:
                if sys.stdout is not None:
                    sys.stdout.write(
                        f"\n=========================================================================================================\n"
                        f"MOLMEM OPTIMIZATION WARNING: Custom source function '{func_name}' is not vectorized!\n"
                        f"=========================================================================================================\n"
                        f"  To maximize execution speed and bypass slow Python element-by-element loops:\n"
                        f"  Avoid using Python ternary checks (e.g., 'A if t < threshold else B') inside your function.\n"
                        f"  Instead, use vectorized NumPy operators (e.g., 'np.where(t < threshold, A, B)').\n"
                        f"=========================================================================================================\n\n"
                    )
                    sys.stdout.flush()
                BSource._warned_funcs.add(func_name)
                
            out = np.zeros_like(t)
            mask = np.zeros(len(t), dtype=bool)
            for t_start, t_end in active_intervals:
                start_idx = np.searchsorted(t, t_start - 1e-15, side='left')
                end_idx = np.searchsorted(t, t_end + 1e-15, side='right')
                mask[start_idx:end_idx] = True
                
            if np.any(mask):
                t_active = t[mask]
                func = self.vFunc
                srcs = self.sources
                try:
                    ufunc = getattr(self, '_cached_ufunc', None)
                    if ufunc is None:
                        ufunc = np.frompyfunc(lambda ti: func(ti, srcs), 1, 1)
                        self._cached_ufunc = ufunc
                    out[mask] = ufunc(t_active).astype(np.float64)
                except Exception:
                    out[mask] = np.array([func(ti, srcs) for ti in t_active], dtype=np.float64)
            return out
            
        return self.vFunc(t, self.sources)

    def getNextEdge(self, t_current):
        min_edge = float('inf')
        t_thresh = t_current + 1e-15
        srcs = self.sources
        src_intervals = self.source_intervals
        len_src_intervals = len(src_intervals) if src_intervals else 0
        for i, src in enumerate(srcs):
            if not hasattr(src, 'getNextEdge'):
                continue
            if i < len_src_intervals and src_intervals[i] is not None:
                intervals = src_intervals[i]
                if isinstance(intervals, tuple):
                    intervals = [intervals]
                for t_start_act, t_end_act in intervals:
                    if t_current < t_end_act:
                        t_search = t_current if t_current > t_start_act else t_start_act
                        edge = src.getNextEdge(t_search)
                        if t_thresh < edge < min_edge and edge <= t_end_act:
                            min_edge = edge
            else:
                edge = src.getNextEdge(t_current)
                if t_thresh < edge < min_edge:
                    min_edge = edge
        edge_pts = self.edge_points
        if edge_pts:
            len_edge_pts = len(edge_pts)
            i_edge = bisect.bisect_right(edge_pts, t_thresh)
            if i_edge < len_edge_pts:
                cand = edge_pts[i_edge]
                if cand < min_edge:
                    min_edge = cand
        return min_edge

    def get_all_edges(self, tStart, tEnd):
        active_intervals = self.get_active_intervals(tStart, tEnd)
        edges = set()
        srcs = self.sources
        num_srcs = len(srcs)
        seg_map = self._segment_source_map
        for t_start_act, t_end_act in active_intervals:
            matched = seg_map.get((t_start_act, t_end_act)) if seg_map is not None else None
            if isinstance(matched, tuple):
                if matched[0] == 'const':
                    continue
                matched_idx = matched[0]
            else:
                matched_idx = matched
            target_sources = (srcs[matched_idx],) if (matched_idx is not None and 0 <= matched_idx < num_srcs) else srcs
            for src in target_sources:
                if not hasattr(src, 'getNextEdge'):
                    continue
                if hasattr(src, 'get_all_edges'):
                    edges.update(src.get_all_edges(t_start_act, t_end_act))
                else:
                    t_search = t_start_act
                    while t_search < t_end_act:
                        edge = src.getNextEdge(t_search)
                        if edge == float('inf') or edge > t_end_act + 1e-12:
                            break
                        edges.add(edge)
                        t_search = edge + 1e-18
        t_min_thresh = tStart - 1e-15
        t_max_thresh = tEnd + 1e-15
        edge_pts = self.edge_points
        if edge_pts:
            i_left = bisect.bisect_left(edge_pts, t_min_thresh)
            i_right = bisect.bisect_right(edge_pts, t_max_thresh)
            edges.update(edge_pts[i_left:i_right])
        return np.array(sorted(edges), dtype=np.float64) if len(edges) > 0 else _EMPTY_ARRAY
