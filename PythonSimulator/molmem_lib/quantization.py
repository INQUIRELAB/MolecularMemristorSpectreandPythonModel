import numpy as np

class WeightQuantizer:
    """
    Simulates a Digital-to-Analog Converter (DAC) for crossbar programming.
    Maps ideal floating-point neural network weights [0.0, 1.0] into discrete 
    hardware levels based on bit-depth, and maps those to the memristor State (n).
    """
    def __init__(self, targetBits=8, minStates=0.0, maxStates=16520.0):
        if targetBits < 1:
            raise ValueError("targetBits must be at least 1")
        self._targetBits = targetBits
        self._minStates = float(minStates)
        self._maxStates = float(maxStates)
        if self._minStates > self._maxStates:
            raise ValueError("minStates must be less than or equal to maxStates")
            
        self._recompute_constants()
        
    def _recompute_constants(self):
        self.numLevels = 1 << self._targetBits
        self.levels_minus_1 = self.numLevels - 1
        self.inv_levels_minus_1 = 1.0 / self.levels_minus_1 if self.levels_minus_1 > 0 else 0.0
        self.state_range = self._maxStates - self._minStates
        self.weight_to_state_scale = self.inv_levels_minus_1 * self.state_range

    @property
    def targetBits(self):
        return self._targetBits

    @targetBits.setter
    def targetBits(self, val):
        self._targetBits = int(val)
        self._recompute_constants()

    @property
    def minStates(self):
        return self._minStates

    @minStates.setter
    def minStates(self, val):
        self._minStates = float(val)
        self._recompute_constants()

    @property
    def maxStates(self):
        return self._maxStates

    @maxStates.setter
    def maxStates(self, val):
        self._maxStates = float(val)
        self._recompute_constants()
        
    def quantizeWeights(self, weights):
        """
        Takes a normalized weight matrix (0.0 to 1.0) and returns the quantized 
        normalized weights.
        """
        levels_m1 = self.levels_minus_1
        inv_levels_m1 = self.inv_levels_minus_1
        if type(weights) is float or type(weights) is int:
            w = float(weights)
            w_clamped = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
            return round(w_clamped * levels_m1) * inv_levels_m1
            
        out = np.clip(weights, 0.0, 1.0, dtype=np.float64)
        out *= levels_m1
        np.rint(out, out=out)
        out *= inv_levels_m1
        return out
        
    def mapToStates(self, weights):
        """
        Takes a normalized weight matrix (0.0 to 1.0) and translates it to 
        the exact target memristor states (n) needed for programming.
        """
        levels_m1 = self.levels_minus_1
        scale = self.weight_to_state_scale
        min_s = self.minStates
        if type(weights) is float or type(weights) is int:
            w = float(weights)
            w_clamped = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
            levels = round(w_clamped * levels_m1)
            return round(levels * scale) if min_s == 0 else round(levels * scale + min_s)
            
        out = np.clip(weights, 0.0, 1.0, dtype=np.float64)
        out *= levels_m1
        np.rint(out, out=out)
        out *= scale
        if min_s != 0:
            out += min_s
        np.rint(out, out=out)
        return out

class OutputQuantizer:
    """
    Simulates an Analog-to-Digital Converter (ADC) reading currents from 
    the crossbar columns and sensing them into discrete output bits.
    """
    def __init__(self, adcBits=8, iMin=0.0, iMax=1.0):
        if adcBits < 1:
            raise ValueError("adcBits must be at least 1")
        if iMin > iMax:
            raise ValueError("iMin must be less than or equal to iMax")
            
        self.adcBits = adcBits
        self.numLevels = 1 << adcBits
        self.iMin = iMin
        self.iMax = iMax
        
        # Precompute constants to optimize vectorized math
        self.levels_minus_1 = self.numLevels - 1
        self.inv_levels_minus_1 = 1.0 / self.levels_minus_1 if self.levels_minus_1 > 0 else 0.0
        self.i_range = self.iMax - self.iMin
        self.inv_i_range = 1.0 / self.i_range if self.i_range > 1e-15 else 0.0
        self.lsb_step = self.inv_levels_minus_1 * self.i_range
        
    def quantizeCurrents(self, currents, returnBits=False):
        """
        Maps a physical current vector to discrete ADC boundaries.
        If returnBits is True, returns the integer level (0 to 2^bits - 1).
        If returnBits is False, returns the quantized current value.
        """
        inv_r = self.inv_i_range
        i_min = self.iMin
        levels_m1 = self.levels_minus_1
        lsb = self.lsb_step
        i_range = self.i_range
        
        if type(currents) is float or type(currents) is int:
            if i_range <= 1e-15:
                return 0 if returnBits else i_min
            c = float(currents)
            val = c * inv_r if i_min == 0.0 else (c - i_min) * inv_r
            norm = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
            lvl = round(norm * levels_m1)
            return lvl if returnBits else (lvl * lsb if i_min == 0.0 else (lvl * lsb + i_min))
            
        if i_range <= 1e-15:
            # Safe safeguard: if range is zero, return iMin levels
            if returnBits:
                return np.zeros_like(currents, dtype=np.int32) if isinstance(currents, np.ndarray) else 0
            return np.full_like(currents, i_min) if isinstance(currents, np.ndarray) else i_min
            
        # Normalize continuous current to [0.0, 1.0] range using fast multiplication
        norm_expr = currents * inv_r if i_min == 0.0 else (currents - i_min) * inv_r
        levels = np.clip(norm_expr, 0.0, 1.0, out=norm_expr if isinstance(norm_expr, np.ndarray) else None, dtype=np.float64)
        levels *= levels_m1
        np.rint(levels, out=levels)
        
        if returnBits:
            # Safely handle any NaN/inf values to prevent NumPy RuntimeWarning on integer cast
            safe_levels = np.nan_to_num(levels, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            return safe_levels.astype(np.int32, copy=False)
            
        # De-normalize back to physical current using fast multiplication
        levels *= lsb
        if i_min != 0.0:
            levels += i_min
        return levels

    def setRange(self, iMin=0.0, iMax=1.0):
        """Sets physical range and updates precomputed scaling factors."""
        f_min = float(iMin)
        f_max = float(iMax)
        if self.iMin == f_min and self.iMax == f_max:
            return
        self.iMin = f_min
        self.iMax = f_max
        i_range = f_max - f_min
        self.i_range = i_range
        self.inv_i_range = 1.0 / i_range if i_range > 1e-15 else 0.0
        inv_l_m1 = self.inv_levels_minus_1
        self.lsb_step = inv_l_m1 * i_range

    def calibrateRange(self, currents):
        """
        Helper method to automatically set i_min and i_max based on 
        a sample set of expected current outputs.
        """
        if isinstance(currents, np.ndarray):
            if currents.size == 0:
                return
            self.setRange(float(currents.min()), float(currents.max()))
        elif type(currents) is float or type(currents) is int:
            c = float(currents)
            self.setRange(c, c)
        else:
            try:
                if len(currents) == 0:
                    return
                self.setRange(float(min(currents)), float(max(currents)))
            except Exception:
                pass

class InputQuantizer:
    """
    Simulates a Digital-to-Time Converter (DTC) for PWM Inference.
    Maps a continuous normalized input vector [0.0, 1.0] into discrete 
    time durations (pulse widths) based on the input bit-depth.
    """
    def __init__(self, inputBits=8):
        if inputBits < 1:
            raise ValueError("inputBits must be at least 1")
            
        self.inputBits = inputBits
        self.numLevels = 1 << inputBits
        self.timeResolution = 1e-9 # 1 ns per LSB
        
        self.levels_minus_1 = self.numLevels - 1
        self.maxPulseWidth = self.timeResolution * self.levels_minus_1

    def quantizeInputs(self, inputs):
        """
        Returns a time array of exact physical pulse widths mapping the input data.
        """
        levels_m1 = self.levels_minus_1
        t_res = self.timeResolution
        if type(inputs) is float or type(inputs) is int:
            w = float(inputs)
            in_clamped = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
            return round(in_clamped * levels_m1) * t_res
            
        out = np.clip(inputs, 0.0, 1.0, dtype=np.float64)
        out *= levels_m1
        np.rint(out, out=out)
        out *= t_res
        return out
