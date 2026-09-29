"""
TensorFlow Integration Layer for the MolMemristor Crossbar Simulator.
Provides a drop-in Keras Dense Layer replacement that executes true analog
physics forward passes through the transient simulator, with Straight-Through
Estimator (STE) gradients for standard backpropagation compatibility.

Also provides system-level hardening utilities for stable Windows execution
under TensorFlow's aggressive C++ runtime and Intel Fortran signal handlers.
"""
import sys
import os
import numpy as np
import importlib.util

from .sys_utils import suppress_c_stderr

# Check if tensorflow is installed before doing any imports or modifications
if importlib.util.find_spec("tensorflow") is None:
    raise ImportError("TensorFlow is not installed in the current environment.")

# Suppress TensorFlow C++ backend logging and driver probe warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

with suppress_c_stderr():
    import tensorflow as tf
    import logging

    # Disable GPU and limit CPU threads for TensorFlow to prevent resource hogging
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass

    tf.get_logger().setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


# --- Global Tracking State for Unified Progress Bars ---
_ACTIVE_EPOCH = 1
_ACTIVE_BATCH = 1
_TOTAL_BATCHES = 1



class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max
        
    def __call__(self, w):
        return tf.clip_by_value(w, self.clip_min, self.clip_max)
        
    def get_config(self):
        return {'clip_min': self.clip_min, 'clip_max': self.clip_max}

class CrossbarDense(tf.keras.layers.Layer):
    """
    A Keras Layer that wraps the MolMemristor transient physics simulator.
    Performs True Analog Physics Forward-Pass with a Straight-Through Estimator Backward-Pass.
    
    Forward:  Programs the physical crossbar to the current Keras weights,
              then simulates real PWM inference pulses through the array.
    Backward: Uses a linear surrogate gradient (STE) so TensorFlow's optimizer
              can differentiate through the black-box physics.
    
    Args:
        units:        Number of output columns (neurons) in the crossbar.
        sim_instance: A fully initialized MolmemSimulator with addCrossbarMatrix already called.
        output_scale: Scalar multiplier to map raw integrated current into standard logit ranges.
                      Typical values: 10000 for 64-input arrays, 100000 for 2-input arrays.
        grad_scale:   Scalar multiplier for the STE surrogate gradient magnitude.
    """
    def __init__(self, units, sim_instance, output_scale=1.0, grad_scale=1.0, fineTune=False, **kwargs):
        super(CrossbarDense, self).__init__(**kwargs)
        self.units = units
        self.sim = sim_instance
        self.output_scale = output_scale
        self.grad_scale = grad_scale
        self.fineTune = fineTune
        self._last_w = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize weights safely within [0.2, 0.8] normalized conductance range to speed up training
        self.w = self.add_weight(shape=(input_dim, self.units),
                                 initializer=tf.keras.initializers.RandomUniform(minval=0.2, maxval=0.8),
                                 constraint=ClipConstraint(0.0, 1.0),
                                 trainable=True,
                                 name='crossbar_weights')
                                 
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='crossbar_bias')

    def call(self, inputs):
        output_scale = self.output_scale
        grad_scale = self.grad_scale
        sim = self.sim
        units = self.units
        
        # The custom gradient decorator allows Keras to differentiate through the black-box physics!
        @tf.custom_gradient
        def crossbar_forward(x_in, w_in):
            
            # --- TRUE PHYSICAL FORWARD PASS ---
            def _hw_forward(x_np, w_np):
                # Ensure weights stay in bounds for the hardware crossbar [0.0, 1.0]
                w_norm = np.clip(w_np, 0.0, 1.0).astype(np.float64, copy=False)
                
                try:
                    global _ACTIVE_BATCH, _TOTAL_BATCHES
                    inv_batches = 1.0 / max(1, _TOTAL_BATCHES)
                    progress = min(100.0, (_ACTIVE_BATCH * inv_batches) * 100.0)
                    bar = '#' * int(progress * 0.2)
                    status = f"Epoch [{bar.ljust(20)}] {progress:3.0f}% | Batch {_ACTIVE_BATCH:02d}/{_TOTAL_BATCHES:02d}"
                    prefix = f"{status} | HW"
                    
                    # 1. Update the physical memristor filaments
                    weights_changed = True
                    if self._last_w is not None and self._last_w.shape == w_norm.shape:
                        if np.array_equal(self._last_w, w_norm):
                            weights_changed = False
                            
                    if weights_changed:
                        # Let the simulator update and pipe its progress native to `prefix`
                        sim.updateCrossbarWeights(w_norm, detailedPrint=False, prefix=prefix, fineTune=self.fineTune)
                        self._last_w = w_norm.copy()
                    
                    # 2. Iterate through batch and perform physics Inference
                    # Mask the Inference prints from the ODE solver
                    old_dp = getattr(sim, 'detailedPrint', True)
                    sim.detailedPrint = False
                    
                    try:
                        stdout = sys.__stdout__
                        if stdout is not None:
                            stdout.write(f"\r  {prefix} [                    ] 0% | Running Batch Inference... ".ljust(95))
                            stdout.flush()
                        
                        # Clip input PWM values to [0.0, 1.0]
                        x_batch = np.clip(x_np, 0.0, 1.0).astype(np.float64, copy=False)
                        
                        # Physical Pass in parallel batch mode! (appendHistory=False prevents RAM buildup)
                        y_bits = sim.passInput(x_batch, detailedPrint=False, appendHistory=False)
                        
                        # Scale raw integrated current into standard NN logit ranges
                        y_out = (y_bits * output_scale).astype(np.float32, copy=False)
                        
                        if stdout is not None:
                            stdout.write(f"\r  {prefix} [{'#'*20}] 100% | [DONE] ".ljust(95))
                            stdout.flush()
                        
                    except BaseException as e:
                        sim.detailedPrint = old_dp
                        if isinstance(e, KeyboardInterrupt):
                            sys.stderr.write("\n[!] Simulation interrupted by user. Terminating process...\n")
                            sys.stderr.flush()
                            try:
                                from .sys_utils import kill_child_processes
                                kill_child_processes()
                            except Exception:
                                pass
                            os.kill(os.getpid(), 9)
                            os._exit(1)
                        print(f"\nHardware Error: {e}")
                        raise e
                    finally:
                        sim.detailedPrint = old_dp
                        
                except Exception as e:
                    print(f"\nOuter Hardware Wrapper Error: {e}")
                    raise e
                    
                return y_out

            # Inject the python crossbar logic into the TensorFlow graph
            y_pred = tf.numpy_function(_hw_forward, [x_in, w_in], tf.float32)
            y_pred.set_shape((None, units))
            
            # --- STRAIGHT-THROUGH ESTIMATOR BACKWARD PASS ---
            # TensorFlow uses exactly this surrogate derivative to learn the true gradients!
            def grad(dy):
                dx = tf.matmul(dy, w_in, transpose_b=True)
                dw = tf.matmul(x_in, dy, transpose_a=True)
                return dx * grad_scale, dw * grad_scale
                
            return y_pred, grad
            
        # Standard Dense Layer Output = X * W + b
        out = crossbar_forward(inputs, self.w) + self.b
        return out


class EpochTracker(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that prints clean, hardware-friendly epoch summaries
    instead of the noisy default progress bar. Designed not to collide with
    the CrossbarDense layer's inline hardware status prints.
    """
    def on_epoch_begin(self, epoch, logs=None):
        global _ACTIVE_EPOCH, _TOTAL_BATCHES
        _ACTIVE_EPOCH = epoch + 1
        _TOTAL_BATCHES = self.params.get('steps', 1)
        print(f"\n[Epoch {_ACTIVE_EPOCH}/{self.params['epochs']}]")
        
    def on_train_batch_begin(self, batch, logs=None):
        global _ACTIVE_BATCH
        _ACTIVE_BATCH = batch + 1

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0.0)
        acc = logs.get('accuracy', 0.0)
        print(f"\r  \u2514\u2500 >> Epoch {epoch+1} Completed | Loss: {loss:.4f} | Accuracy: {acc:.4f} <<{' '*80}", flush=True)
