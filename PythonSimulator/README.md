# Molmem Simulator: Molecular Memristor Simulation Platform

A high-performance, physics-based numerical simulation platform for continuous-state molecular memristors, dense nanoscale crossbar arrays, tiled neuromorphic accelerators, and mixed-signal circuit testbenches.

---

## Table of Contents
1. [Architecture & Platform Overview](#1-architecture--platform-overview)
2. [Quick Start & Runtime Environment](#2-quick-start--runtime-environment)
3. [Comprehensive Testbench API Reference](#3-comprehensive-testbench-api-reference)
   - [Core Simulation Engine (`MolmemSimulator`)](#31-core-simulation-engine-molmemsimulator)
   - [Modular Voltage Sources (`molmem_lib.sources`)](#32-modular-voltage-sources-molmem_libsources)
   - [Device Models & Parameter Presets (`molmem_lib.devices`)](#33-device-models--parameter-presets-molmem_libdevices)
   - [Waveform Visualization & Analytics (`MolmemPlotter`)](#34-waveform-visualization--analytics-molmemplotter)
   - [Tiled Crossbar Subsystem (`MolmemTiledSimulator`)](#35-tiled-crossbar-subsystem-molmemtiledsimulator)
   - [Parallel Multi-Array Dispatch (`updateCrossbarsParallel`)](#36-parallel-multi-array-dispatch-updatecrossbarsparallel)
   - [Hardware Quantization Converters (`molmem_lib.quantization`)](#37-hardware-quantization-converters-molmem_libquantization)
   - [Empirical Statistical Noise Engine (`NoiseEngine`)](#38-empirical-statistical-noise-engine-noiseengine)
   - [Machine Learning Hardware Integration (`CrossbarDense`, `EpochTracker`)](#39-machine-learning-hardware-integration-crossbardense-epochtracker)
   - [Closed-Loop Array Programmer (`CrossbarProgrammer`)](#310-closed-loop-array-programmer-crossbarprogrammer)
   - [Global Runtime Controls & Context Managers](#311-global-runtime-controls--context-managers)
4. [Representative Testbench Walkthroughs](#4-representative-testbench-walkthroughs)
   - [Walkthrough 1: Dynamic SET/RESET Cycling & Thermal Relaxation](#walkthrough-1-dynamic-setreset-cycling--thermal-relaxation)
   - [Walkthrough 2: N-by-M Nanoscale Crossbar Transient Simulation](#walkthrough-2-n-by-m-nanoscale-crossbar-transient-simulation)
   - [Walkthrough 3: Addressable Tiled Crossbar Array Acceleration](#walkthrough-3-addressable-tiled-crossbar-array-acceleration)
   - [Walkthrough 4: Multi-Crossbar Thread-Resource Parallel Programming](#walkthrough-4-multi-crossbar-thread-resource-parallel-programming)
   - [Walkthrough 5: TensorFlow Hardware-in-the-Loop Gradient Training](#walkthrough-5-tensorflow-hardware-in-the-loop-gradient-training)
5. [Brief Catalog of Auxiliary & Internal API Commands](#5-brief-catalog-of-auxiliary--internal-api-commands)

---

## 1. Architecture & Platform Overview

`molmem_lib` simulates molecular redox and conformational switching memristors based on experimentally validated physical kinetics rather than abstract empirical curve fits.

### Physical Foundations
* **Continuous Kinetic State Engine**: Integrates the internal molecular population state variable ($n \in [0, 33040]$), representing the continuous migration and conformational alignment of redox-active molecular cores under electric field stress.
* **Non-Linear Sinh Conduction**: Models field-assisted hopping and quantum tunneling through molecular monolayers:
  $$I(V, n) = I_{\text{scale}}(n) \cdot \sinh\left(\frac{V_{\text{junction}}}{V_{\text{scale}}(n)}\right)$$
* **Newton-Raphson Contact Series Resistance**: Explicitly resolves the non-linear transcendental voltage divider formed between the internal molecular junction and the metallic parasitic contact resistance ($R_c$) at every simulation time step.
* **Coupled Self-Heating & Exponential Relaxation**: Dynamically tracks junction self-heating via instantaneous Joule dissipation ($P = I \cdot V_{\text{junction}}$), coupled to an Arrhenius reaction rate boost and thermal relaxation back to ambient temperature ($T_{\text{amb}} = 298.15\text{ K}$) with thermal time constant $\tau_{\text{th}} = 127.8\text{ ns}$.
* **Empirical Noise Channels**: Directly injects experimentally extracted non-idealities:
  * *Device-to-Device (D2D)* spatial mismatch via empirical inverse-CDF quantile mapping.
  * *Temporal Read Noise* incorporating Johnson-Nyquist thermal white noise floor and Hooge $1/f$ flicker noise.
  * *Cycle-to-Cycle Write Noise* modeling Langevin stochastic ionic hopping.

### Execution Backends & Memory Pipeline
* **CPU Backend (Numba JIT + Intel TBB)**: Structures device states in C-contiguous Structure-of-Arrays (`_soa_*`) memory layouts. Adaptive predictor-corrector nodal loops execute without Python GIL overhead at near-native C speeds.
* **GPU Backend (PyTorch CUDA / AMD ROCm HIP)**: Massively parallel matrix-vector pulse propagation and instant cell current evaluation using native tensor math or compiled C++/HIP extensions (`molmem_cuda_solver.so`).
* **Streaming History Pipeline**: High-throughput asynchronous telemetry logging engine (`history.py`). Flushes transient waveforms into pre-allocated memory-mapped SSD ring buffers, allowing 50,000,000+ simulation steps while maintaining physical host RAM usage below 100 MB.

---

## 2. Quick Start & Runtime Environment

### Installation & Prerequisites
The library operates on Python 3.10+ and requires standard scientific packages:

```bash
pip install numpy scipy matplotlib numba torch
```

To enable the machine learning integration layer, install TensorFlow:
```bash
pip install tensorflow
```

### Dynamic CLI Runtime Flags
The simulator intercepts runtime flags directly from `sys.argv` upon initial import:
* `-silent` / `--silent`: Completely suppresses startup hardware banners and operational console outputs.
* `-cpu` / `--cpu-only`: Enforces CPU-only execution, disabling GPU driver detection and CUDA/HIP initializations.
* `-gpu` / `--force-gpu`: Enforces GPU execution across all supported operations.
* `-profile` / `--profile`: Enables fine-grained function-level profiling and hardware affinity counters.
* `-profile-ssd`: Activates detailed throughput and bandwidth instrumentation on the memory-mapped SSD logging pipeline.
* `-debug-gpu`: Enables synchronous GPU execution and asserts CUDA/HIP launch blocking (`CUDA_LAUNCH_BLOCKING=1`).
* `--clear-cache`: Purges pre-compiled LLVM JIT bytecode, hardware calibration maps, and affinity profiles.

---

## 3. Comprehensive Testbench API Reference

This section provides exhaustive reference documentation for every single API command and class method utilized by circuit-level, software-level, and hardware validation testbenches throughout the codebase.

---

### 3.1 Core Simulation Engine (`MolmemSimulator`)

Located in `molmem_lib.simulator.MolmemSimulator`.

```python
from molmem_lib import MolmemSimulator
```

#### `MolmemSimulator.__init__(instanceName="crossbarArray", device='auto', backend='auto', vram_fraction=0.8)`
Instantiates the transient simulation manager.

* **Parameters**:
  * `instanceName` (*str*, default=`"crossbarArray"`): Unique identifier label used for logging, diagnostic console output, and memory profiling.
  * `device` (*str*, default=`'auto'`): Compute hardware target. Options include `'auto'`, `'cpu'`, `'cuda'`, `'cuda:0'`, or explicit PyTorch device strings. In interactive notebooks (Jupyter, Colab), defaults safely to `'cpu'`.
  * `backend` (*str*, default=`'auto'`): Solver backend implementation. Options include `'auto'`, `'numba'` (CPU JIT), or `'pytorch'` / `'torch'` (GPU tensor acceleration).
  * `vram_fraction` (*float*, default=`0.8`): Maximum proportion of available GPU VRAM allocated for transient pulse schedules and batch buffers.

---

#### `sim.addCrossbarMatrix(...)`
```python
sim.addCrossbarMatrix(
    matrix=None, rows=1, cols=1, deviceClass=None, device_type="default",
    weightBits=None, inputBits=None, outputBits=None,
    vWrite=0.9, vErase=-0.75, max_vWrite=5.0, max_pw=320e-9,
    detailedPrint=None, architecture='1T1R', UpdatesPer="Column",
    force_device_init=True, multiThread=None,
    write_noise_intensity=0.0, read_noise_intensity=0.0, d2d_variation_intensity=0.0,
    ignore_safety_cap=False, **kwargs
)
```
Configures an $R \times C$ crossbar array grid within the nodal circuit netlist.

* **Parameters**:
  * `matrix` (*np.ndarray*, optional): Pre-existing weight or state matrix defining array geometry.
  * `rows` (*int*, default=`1`): Number of wordlines (horizontal input rows).
  * `cols` (*int*, default=`1`): Number of bitlines (vertical output columns).
  * `deviceClass` (*class*, optional): Class type used for individual memristors (defaults to `MolMemristor`).
  * `device_type` (*str* or *dict*, default=`"default"`): Device parameter profile name (`"default"`, `"ru_azo"`) or a custom dictionary obtained from `get_device_config()`.
  * `weightBits` (*int*, optional): Digital DAC bit-depth for synaptic weights (e.g. `8` or `14`). When specified, auto-instantiates `WeightQuantizer`.
  * `inputBits` (*int*, optional): Digital DTC bit-depth for input pulse widths (e.g. `8` or `14`). Auto-instantiates `InputQuantizer`.
  * `outputBits` (*int*, optional): Digital ADC bit-depth for sensing integrated column current. Auto-instantiates `OutputQuantizer`.
  * `vWrite` (*float*, default=`0.9`): Nominal potentiation write pulse amplitude in Volts ($V$).
  * `vErase` (*float*, default=`-0.75`): Nominal depression erase pulse amplitude in Volts ($V$).
  * `max_vWrite` (*float*, default=`5.0`): Hard voltage limit cap for closed-loop programming pulses in Volts ($V$).
  * `max_pw` (*float*, default=`320e-9`): Hard pulse duration limit cap in seconds ($s$).
  * `detailedPrint` (*bool*, optional): Enables verbose diagnostic matrix instantiation logs.
  * `architecture` (*str*, default=`'1T1R'`): Crossbar cell architecture. Supports `'1T1R'` (isolated access transistors) and `'passive'` (sneak-path non-linear diode/memristor mesh).
  * `UpdatesPer` (*str*, default=`"Column"`): Programming addressing mode (`"Column"`, `"Row"`, or `"Cell"`).
  * `force_device_init` (*bool*, default=`True`): Enforces clean physical initialization of device internal variables.
  * `multiThread` (*bool*, optional): Overrides automatic CPU multi-threading detection.
  * `write_noise_intensity` (*float*, default=`0.0`): Cycle-to-cycle write noise scaling ($0.0 = \text{deterministic}$, $1.0 = \text{nominal experimental variance}$).
  * `read_noise_intensity` (*float*, default=`0.0`): Temporal read noise scaling ($0.0 = \text{disabled}$, $1.0 = \text{nominal thermal + flicker}$).
  * `d2d_variation_intensity` (*float*, default=`0.0`): Spatial device-to-device parameter spread scaling.
* **Returns**: None.

---

#### `sim.addMemristor(topNode, bottomNode, deviceInstance)`
Directly inserts an individual discrete memristor device instance into the circuit netlist between two specified nodes.

* **Parameters**:
  * `topNode` (*str* or *int*): Anode circuit node identifier.
  * `bottomNode` (*str* or *int*): Cathode circuit node identifier.
  * `deviceInstance` (*MolMemristor*): An instantiated molecular memristor object.
* **Returns**: None.

---

#### `sim.addDcSource(row=None, col=None, node=None, voltage=0.0)`
Attaches an ideal DC constant voltage source to a crossbar row, column, or arbitrary circuit node.

* **Parameters**:
  * `row` (*int*, optional): 1-indexed wordline row number.
  * `col` (*int*, optional): 1-indexed bitline column number.
  * `node` (*str* or *int*, optional): Explicit circuit node name.
  * `voltage` (*float*, default=`0.0`): Constant DC bias in Volts ($V$).
* **Returns**: None.

---

#### `sim.addPulseSource(row=None, col=None, node=None, amplitude=0.0, period=0.0, dutyCycle=0.5, delay=0.0, numPulses=None)`
Attaches a periodic rectangular pulse train source to a row, column, or node.

* **Parameters**:
  * `row` / `col` / `node`: Target connection coordinate (1-indexed).
  * `amplitude` (*float*): Pulse peak voltage amplitude in Volts ($V$).
  * `period` (*float*): Pulse repetition interval in seconds ($s$).
  * `dutyCycle` (*float*, default=`0.5`): High-time fraction ($t_{\text{high}} = \text{period} \times \text{dutyCycle}$).
  * `delay` (*float*, default=`0.0`): Initial onset delay in seconds ($s$).
  * `numPulses` (*int*, optional): Maximum pulse count before clamping output to 0 V.
* **Returns**: None.

---

#### `sim.addPwlSource(row=None, col=None, node=None, t_points=(), v_points=())`
Attaches an arbitrary Piecewise Linear (PWL) continuous waveform source.

* **Parameters**:
  * `row` / `col` / `node`: Target connection coordinate.
  * `t_points` (*Sequence[float]*): Strictly monotonic sequence of timestamps in seconds ($s$).
  * `v_points` (*Sequence[float]*): Corresponding voltage levels in Volts ($V$).
* **Returns**: None.

---

#### `sim.addBSource(row=None, col=None, node=None, vFunc=None, sources=(), edge_points=(), source_intervals=None)`
Attaches an adaptive behavioral voltage source evaluated dynamically at runtime via a Python callable.

* **Parameters**:
  * `row` / `col` / `node`: Target connection coordinate.
  * `vFunc` (*Callable[[float, Sequence[BaseSource]], float]*): Dynamic function taking `(t, sources)` and returning instantaneous voltage in Volts ($V$).
  * `sources` (*Sequence[BaseSource]*): Tuple or list of underlying reference sources passed into `vFunc`.
  * `edge_points` (*Sequence[float]*): Explicit timestamps where sharp switching transitions or non-differentiable slopes occur. Informs the adaptive solver to restrict time-steps and eliminate overshoot.
  * `source_intervals` (*Sequence[Tuple[float, float]]*, optional): Active time windows outside of which source evaluation is bypassed.
* **Returns**: None.

---

#### `sim.addVoltageSource(node=None, sourceObj=None, row=None, col=None)` / `sim.addDynamicSource(...)`
Directly connects a pre-instantiated `BaseSource` subclass (`DcSource`, `PulseSource`, `PwlSource`, or `BSource`) to the specified target.

---

#### `sim.clearVoltageSources()`
Removes all attached DC, Pulse, PWL, and behavioral voltage sources from all rows, columns, and nodes, returning external terminals to high-impedance floating states.

---

#### `sim.run(tEnd, min_dt=1e-12, tol=1e-3, title=None, verbose=True, recordHistory=None, recordStepTiming=False, appendHistory=False, detailed=False)`
Executes transient ODE integration from current simulation time up to `tEnd`.

* **Parameters**:
  * `tEnd` (*float*): Stop time for transient integration in seconds ($s$).
  * `min_dt` (*float*, default=`1e-12`): Minimum allowable solver integration time-step ($s$).
  * `tol` (*float*, default=`1e-3`): Predictor-corrector local truncation error tolerance.
  * `title` (*str*, optional): Label displayed in terminal progress bars.
  * `verbose` (*bool*, default=`True`): Controls terminal progress display.
  * `recordHistory` (*bool*, *str*, or *list*, default=`None`): Telemetry recording selection. Pass `None` or `'all'` for full logging, `False` for zero logging, or a list of channels (`['v', 'i', 'g', 'state', 'temp', 'f22']`).
  * `recordStepTiming` (*bool*, default=`False`): Instruments per-step solver wall-clock durations for performance profiling.
  * `appendHistory` (*bool*, default=`False`): When `True`, appends new telemetry to existing memory buffers instead of overwriting from time $t = 0$.
  * `detailed` (*bool*, default=`False`): Enables granular percentage and iteration speed status bars.
* **Returns**: None.

---

#### Telemetry Getters
All getters return 1D NumPy arrays (`np.ndarray`) mapped across recorded transient time steps:

* `sim.getTime()`: Returns simulation timestamps in seconds ($s$).
* `sim.getVoltage(node=None, row=None, col=None)`: Returns nodal voltage in Volts ($V$).
* `sim.getCurrent(row=None, col=None)`: Returns instantaneous device terminal current in Amperes ($A$).
* `sim.getState(row=None, col=None)`: Returns the internal molecular population state variable ($n$, dimensionless integer/float count).
* `sim.getF22(row=None, col=None)`: Returns the normalized physical conductance state ($f_{22} \in [0.0, 1.0]$).
* `sim.getConductance(row=None, col=None)`: Returns instantaneous small-signal chord conductance in Siemens ($S$).
* `sim.getTemperature(row=None, col=None)`: Returns localized molecular junction temperature in Kelvin ($K$).

---

#### `sim.updateCrossbarWeights(...)`
```python
sim.updateCrossbarWeights(
    targetWeightsNormalized, detailedPrint=None, prefix="",
    fineTune=True, recordHistory=False, silentUpdate=False
)
```
Executes closed-loop physical pulse programming to set crossbar conductances to match a normalized weight matrix $W \in [0.0, 1.0]$.

* **Parameters**:
  * `targetWeightsNormalized` (*np.ndarray*): 2D matrix of shape `(rows, cols)` with values normalized in $[0.0, 1.0]$.
  * `detailedPrint` (*bool*, optional): Enables verbose terminal breakdown of Pass 1 and Pass 2 progress.
  * `prefix` (*str*, default=`""`): String prefix prepended to console progress indicators.
  * `fineTune` (*bool*, default=`True`): When `True`, executes Pass 2 correction cycle to eliminate $V/2$ sneak-path drift. When `False`, stops after Pass 1 for rapid execution.
  * `recordHistory` (*bool*, default=`False`): When `False`, automatically purges internal pulse history to avoid RAM creep across training epochs.
  * `silentUpdate` (*bool*, default=`False`): Suppresses all terminal outputs.
* **Returns**: None. Physical conductance can be verified immediately via `readCrossbarMatrix()`.

---

#### `sim.programWeights(targetWeightsNormalized, parallel=True, detailedPrint=False, **kwargs)`
High-level convenience alias for `updateCrossbarWeights()`. Auto-provisions 8-bit quantizers and a `CrossbarProgrammer` instance if the matrix was instantiated without bit-depth configurations.

---

#### `sim.passInput(xNormalized, detailedPrint=True, appendHistory=False, stream=None, return_torch=False)`
Executes physical analog matrix-vector inference. Converts normalized input vectors into PWM pulse durations, simulates column integration through the crossbar, digitizes current via the ADC, and returns the normalized product $Y = X \cdot W$.

* **Parameters**:
  * `xNormalized` (*np.ndarray* or *torch.Tensor*): 1D input vector of shape `(rows,)` or 2D batch of shape `(batch_size, rows)` with values in $[0.0, 1.0]$.
  * `detailedPrint` (*bool*, default=`True`): Prints inference execution metrics.
  * `appendHistory` (*bool*, default=`False`): Records transient pulse waveforms (valid only for 1D single-vector inputs).
  * `stream` (*torch.cuda.Stream*, optional): PyTorch CUDA/HIP stream for asynchronous execution.
  * `return_torch` (*bool*, default=`False`): When `True`, returns output as a GPU PyTorch Tensor, bypassing host DMA transfers.
* **Returns**: Output array/tensor of shape `(cols,)` or `(batch_size, cols)`.

---

#### `sim.readCrossbarMatrix(detailedPrint=True, purpose="Diagnostic Matrix State Read", returnAnalog=False)`
Performs non-destructive physical reading of all crossbar cells using a small sub-threshold read bias ($V_{\text{read}} = 0.1\text{ V}$).

* **Parameters**:
  * `detailedPrint` (*bool*, default=`True`): Prints formatted matrices to terminal.
  * `purpose` (*str*, optional): Header description displayed in terminal logs.
  * `returnAnalog` (*bool*, default=`False`): When `True`, returns unquantized analog conductance values.
* **Returns**:
  * `gMatrixQuantized` (*np.ndarray*): 2D matrix of physical conductances in Siemens ($S$).
  * `iMatrixBits` (*np.ndarray*): 2D integer array of digitized ADC output codes.

---

#### State & Thermal Management Methods
* `sim.resetDeviceStates(n=0.0, modeState=0.0, scaleFactor=1.0, T=298.15)`: Resets all devices in the array to initial state population $n$, clearing previous programming.
* `sim.relaxDeviceTemperatures(T_amb=298.15)`: Exponentially decays internal device temperatures back to ambient temperature ($298.15\text{ K}$).
* `sim.clearHistory(force_gc=False)`: Releases memory-mapped history files and arrays without modifying device states.
* `sim.clearInternalCaches(force_gc=False)`: Flushes JIT schedule buffers, nodal maps, and array caches.
* `sim.sync_to_cpu()`: Enforces memory barrier synchronization, moving GPU VRAM state tensors into host Python device objects.

---

### 3.2 Modular Voltage Sources (`molmem_lib.sources`)

Located in `molmem_lib.sources`. All sources derive from `BaseSource`.

```python
from molmem_lib.sources import BaseSource, DcSource, PulseSource, PwlSource, BSource
```

#### `BaseSource(node)`
Abstract base class for circuit excitation sources.
* `src.getVoltage(t)`: Returns scalar voltage in Volts ($V$) at time $t$.
* `src.getNextEdge(t_current)`: Returns timestamp of the next transition edge after `t_current`.
* `src.get_all_edges(tStart, tEnd)`: Returns sorted list of all discrete transition timestamps within the window $[t_{\text{start}}, t_{\text{end}}]$.

#### `DcSource(node, voltage)`
* `node`: Connected node identifier.
* `voltage` (*float*): Constant potential in Volts ($V$).

#### `PulseSource(node, amp, period, width, delay=0.0, max_pulses=None)`
* `amp` (*float*): Pulse peak amplitude ($V$).
* `period` (*float*): Pulse period ($s$).
* `width` (*float*): Pulse duration ($s$).
* `delay` (*float*, default=`0.0`): Onset delay ($s$).
* `max_pulses` (*int*, optional): Maximum pulse count.

#### `PwlSource(node, t_points, v_points)`
* `t_points` (*Sequence[float]*): Array of time points in seconds ($s$).
* `v_points` (*Sequence[float]*): Array of corresponding voltage points in Volts ($V$).

#### `BSource(node, vFunc, sources, edge_points=None, source_intervals=None)`
* `vFunc` (*Callable[[float, Sequence[BaseSource]], float]*): Behavioral mathematical function $V(t) = f(t, \text{sources})$.
* `sources` (*Sequence[BaseSource]*): Reference input sources passed to `vFunc`.
* `edge_points` (*Sequence[float]*, optional): Discontinuity points for solver step adaptation.

---

### 3.3 Device Models & Parameter Presets (`molmem_lib.devices`)

Located in `molmem_lib.devices` and `molmem_lib.device`.

```python
from molmem_lib import get_device_config, MolMemristor
```

#### `get_device_config(device_type="default")`
Retrieves a validated, physics-calibrated parameter dictionary for the specified device preset.

* **Available Presets**:
  * `"default"`: Calibrated baseline molecular redox memristor ($n_{\text{max}} = 33040$, $V_{\text{ref,pos}} = 0.9\text{ V}$, $V_{\text{ref,neg}} = -0.75\text{ V}$, $R_c = 25\ \Omega$).
  * `"ru_azo"`: Experimental Ruthenium-azo molecular complex memristor calibrated against multi-cycle electrical characterization sweeps.
* **Returns**: Deep-copied dictionary containing physical constants (`kappa`, `alpha`, `vTh`, `rC`, `kDischarge`, `rateAsymmetry`, etc.). Modifying this dictionary allows creating customized device physics configurations.

#### `MolMemristor(topNode, bottomNode, device_type="default", ...)`
The continuous-kinetic molecular memristor instance modeling state ODEs, contact resistance, and Joule heating.

---

### 3.4 Waveform Visualization & Analytics (`MolmemPlotter`)

Located in `molmem_lib.plotter.MolmemPlotter`.

```python
from molmem_lib import MolmemPlotter
```

#### `MolmemPlotter.__init__(sim)`
Initializes the plotting coordinator attached to a `MolmemSimulator` instance.

#### `plotter.createFigure(name, title=None, figsize=(10, 4), subplots=1)`
Provisions a Matplotlib figure managed under an identifier key.

* **Parameters**:
  * `name` (*str*): Unique string lookup identifier.
  * `title` (*str*, optional): Super-title displayed above all subplots.
  * `figsize` (*Tuple[float, float]*, default=`(10, 4)`): Figure dimensions in inches `(width, height)`.
  * `subplots` (*int* or *Tuple[int, int]*, default=`1`): Subplot layout grid. Supports single integers (e.g. `subplots=3` for a $3 \times 1$ vertical stack) or grid tuples (e.g. `subplots=(4, 4)` for a $4 \times 4$ crossbar matrix).

#### `plotter.addWaveform(...)`
```python
plotter.addWaveform(
    name, dataType, row=None, col=None, node=None,
    subplotIdx=0, twinx=False, customData=None, **kwargs
)
```
Extracts simulation history and renders a waveform onto the target figure.

* **Parameters**:
  * `name` (*str*): Target figure identifier.
  * `dataType` (*str*): Telemetry channel selector:
    * `'Voltage'`: Nodal or line voltage (auto-scaled to V or mV).
    * `'Current'`: Device current (auto-scaled to A, mA, $\mu$A, or nA).
    * `'Pulses'`: Molecular population state count $n$.
    * `'State'` or `'f22'`: Dimensionless conductance state $f_{22}$.
    * `'Conductance'`: Chord conductance (auto-scaled to S, mS, $\mu$S, or nS).
    * `'Temperature'` or `'Temp'`: Junction temperature in Kelvin ($K$).
    * `'Custom'`: External user data provided via `customData`.
  * `row` (*int*, optional): 1-indexed row coordinate.
  * `col` (*int*, optional): 1-indexed column coordinate.
  * `node` (*str* or *int*, optional): Circuit node name.
  * `subplotIdx` (*int* or *Tuple[int, int]*, default=`0`): 0-indexed subplot position. Can be a flat integer or `(r, c)` grid index.
  * `twinx` (*bool*, default=`False`): When `True`, generates and plots onto an overlay twin y-axis.
  * `customData` (*np.ndarray*, optional): Array of y-values when `dataType='Custom'`.
  * `**kwargs`: Standard Matplotlib styling arguments (`color`, `label`, `linestyle`, `linewidth`, `alpha`).

#### `plotter.formatAxis(name, subplotIdx=0, title=None, xlabel='Auto', ylabel=None, xlim=None, ylim=None, grid=True, legend=True, logy=False)`
Configures labels, bounds, grids, and scales for a given subplot. Setting `xlabel='Auto'` automatically resolves physical time units (s, ms, $\mu$s, ns).

#### `plotter.saveFigure(name, savePath)`
Renders the figure to disk (supports `.png`, `.jpg`, `.pdf`, `.svg`). Automatically creates missing parent directories.

#### `plotter.renderFigure(name)`
Explicitly renders the figure canvas into memory (useful in interactive notebooks).

#### `plotter.showAll()`
Displays all open figure windows interactively via the configured GUI backend.

---

### 3.5 Tiled Crossbar Subsystem (`MolmemTiledSimulator`)

Located in `molmem_lib.tiled_simulator.MolmemTiledSimulator`. Designed for scaled-up macro architectures by decomposing large matrices into an array of smaller physical tiles (e.g. $64 \times 64$ divided into four $32 \times 32$ tiles).

```python
from molmem_lib import MolmemTiledSimulator
```

#### `MolmemTiledSimulator.__init__(instanceName="TiledArray", device='auto', backend='auto', vram_fraction=0.8)`
Instantiates the multi-tile fabric coordinator.

#### `tiled_sim.addTiledCrossbarMatrix(...)`
```python
tiled_sim.addTiledCrossbarMatrix(
    total_rows, total_cols, tile_rows, tile_cols,
    device_type="default", weightBits=None, inputBits=None, outputBits=None,
    vWrite=0.9, vErase=-0.75, max_vWrite=5.0, max_pw=320e-9,
    detailedPrint=True, architecture='1T1R', UpdatesPer="Column",
    force_device_init=True, write_noise_intensity=0.0,
    read_noise_intensity=0.0, d2d_variation_intensity=0.0
)
```
Partitions a large global crossbar of size `total_rows x total_cols` into independent sub-tiles of dimension `tile_rows x tile_cols`.

#### `tiled_sim.updateTiledCrossbarWeights(targetWeightsNormalized, detailedPrint=True, prefix="", fineTune=True, recordHistory=False, silentUpdate=False, **kwargs)`
Dispatches programming pulses across all constituent sub-tiles in parallel to realize the complete global weight matrix.

#### `tiled_sim.updateTileCrossbarWeights(tile_r, tile_c, targetWeightsNormalized, detailedPrint=True, prefix="", fineTune=True, recordHistory=False, silentUpdate=False, **kwargs)`
Selectively reprograms a single sub-tile at grid coordinate `(tile_r, tile_c)` without altering conductances in neighboring tiles.

#### `tiled_sim.passTiledCrossbarInput(xNormalized, detailedPrint=True, appendHistory=False, stream=None, return_torch=False, **kwargs)`
Executes tiled analog inference. Distributes input vector slices across tile rows, sums bitline currents down tile columns, and normalizes output.

#### `tiled_sim.readTiledCrossbarMatrix(detailedPrint=True, purpose=None, **kwargs)`
Assembles and returns the global conductance matrix aggregated across all physical sub-tiles.

---

### 3.6 Parallel Multi-Array Dispatch (`updateCrossbarsParallel`)

Located in `molmem_lib.simulator.updateCrossbarsParallel`.

```python
from molmem_lib import updateCrossbarsParallel
```

#### `updateCrossbarsParallel(simulators, weight_matrices, detailedPrint=True, prefix="", fineTune=True, recordHistory=False)`
Simultaneously programs multiple isolated, disjoint `MolmemSimulator` instances (e.g. differential positive/negative weight arrays or multiple layers in a deep network) using dynamic thread-pool allocation and dedicated GPU compute streams.

* **Parameters**:
  * `simulators` (*Sequence[MolmemSimulator]*): List of independent simulator instances.
  * `weight_matrices` (*Sequence[np.ndarray]*): List of corresponding normalized weight matrices matching each simulator's geometry.
  * `detailedPrint` (*bool*, default=`True`): Controls parallel progress bar reporting.
  * `prefix` (*str*, default=`""`): Prefix string for terminal status indicators.
  * `fineTune` (*bool*, default=`True`): Enables Pass 2 correction cycle across all arrays.
  * `recordHistory` (*bool*, default=`False`): Purges pulse history upon completion to preserve RAM.
* **Returns**: None.

---

### 3.7 Hardware Quantization Converters (`molmem_lib.quantization`)

Located in `molmem_lib.quantization`.

```python
from molmem_lib import WeightQuantizer, InputQuantizer, OutputQuantizer
```

#### `WeightQuantizer(targetBits=8, minStates=0.0, maxStates=16520.0)`
Simulates a programming DAC translating continuous neural weights into discrete device states ($n$).
* `wq.quantizeWeights(weights)`: Clamps input array to $[0.0, 1.0]$ and rounds to the nearest discrete DAC step.
* `wq.mapToStates(weights)`: Maps normalized weights into discrete integer molecular states $n \in [n_{\text{min}}, n_{\text{max}}]$.

#### `InputQuantizer(inputBits=8)`
Simulates a Digital-to-Time Converter (DTC) for PWM inference.
* `inQ.quantizeInputs(inputs)`: Converts continuous inputs $[0.0, 1.0]$ into physical pulse widths ($s$) based on 1 ns LSB resolution ($t_{\text{pulse}} = \text{level} \times 1\text{ ns}$).

#### `OutputQuantizer(adcBits=8, iMin=0.0, iMax=1.0)`
Simulates a column sense-amplifier ADC digitizing continuous integrated currents.
* `adc.quantizeCurrents(currents, returnBits=False)`: Digitizes current into discrete levels. If `returnBits=True`, returns integer codes ($0$ to $2^{\text{bits}}-1$); if `False`, returns quantized analog Amperes ($A$).
* `adc.setRange(iMin, iMax)`: Sets physical full-scale current conversion boundaries.
* `adc.calibrateRange(currents)`: Automatically snaps `iMin` and `iMax` to the dynamic range of a sample current matrix.

---

### 3.8 Empirical Statistical Noise Engine (`NoiseEngine`)

Located in `molmem_lib.noise.NoiseEngine`.

```python
from molmem_lib import NoiseEngine
```

#### `NoiseEngine.sample_d2d_factors(shape, intensity=0.0)`
Samples spatial device-to-device parameter perturbations from the empirical inverse-CDF quantile table. Returns fractional multipliers: $P_{\text{device}} = P_{\text{nom}} \times (1.0 + \text{intensity} \times \delta)$.

#### `NoiseEngine.inject_read_noise_cpu(i_analog, v_read, g_min, g_max, intensity=0.0)`
Injects state-dependent Johnson-Nyquist thermal floor and Hooge $1/f$ flicker noise into a NumPy current array.

#### `NoiseEngine.inject_read_noise_torch(i_cell_tensor, v_read, g_min, g_max, intensity=0.0)`
Performs GPU-accelerated piecewise linear interpolation and noise injection directly on a PyTorch CUDA tensor without host PCIe transfers.

#### `NoiseEngine.inject_write_noise(delta_state, state_norm, intensity=0.0)`
Perturbs incremental state changes ($\Delta n$) with Langevin ionic hopping variance during programming pulses.

---

### 3.9 Machine Learning Hardware Integration (`CrossbarDense`, `EpochTracker`)

Located in `molmem_lib.tf_wrapper`.

```python
from molmem_lib import CrossbarDense, EpochTracker, suppress_c_stderr
```

#### `CrossbarDense(units, sim_instance, output_scale=1.0, grad_scale=1.0, fineTune=False, **kwargs)`
A Keras-compatible dense layer performing physical analog forward inference with Straight-Through Estimator (STE) backpropagation.

* **Parameters**:
  * `units` (*int*): Number of output neurons (bitline columns).
  * `sim_instance` (*MolmemSimulator*): Pre-configured simulator instance matching layer input and output dimensions.
  * `output_scale` (*float*, default=`1.0`): Scalar multiplier mapping integrated physical current into neural network logit scale.
  * `grad_scale` (*float*, default=`1.0`): Multiplier on surrogate backpropagation gradients.
  * `fineTune` (*bool*, default=`False`): Enables Pass 2 closed-loop correction during weight updates.

#### `EpochTracker()`
Custom Keras callback providing clean terminal status reports per epoch that do not collide with inline crossbar hardware progress indicators.

#### `suppress_c_stderr()`
Context manager intercepting low-level OS file descriptors (fd 2) to eliminate C++ runtime warnings (e.g. TensorFlow driver probing or Intel OpenMP duplicate runtime notifications).

---

### 3.10 Closed-Loop Array Programmer (`CrossbarProgrammer`)

Located in `molmem_lib.programmer.CrossbarProgrammer`.

```python
from molmem_lib import CrossbarProgrammer
```

#### `CrossbarProgrammer.__init__(simulator, vWrite=0.9, vErase=-0.75, pulseWidthPot=80e-9, pulseWidthDep=60e-9, period=160e-9, max_vWrite=5.0, max_pw=320e-9)`
Initializes the inverse pulse solver for a target simulator.

#### `programmer.calibrate(maxPulses=None, maxLinearState=None)`
Executes characterization sweeps on a baseline cell to build the forward and inverse response lookup surfaces (`stateMapPot`, `stateMapDep`, `gMapG`, `gMapN`).

#### `programmer.programWeightsParallel(targetStates, isCorrectionPass=False, tolerance=0.005, recordHistory=False)`
Computes optimal pulse counts and amplitudes, then executes column-parallel pulse programming across the array.

---

### 3.11 Global Runtime Controls & Context Managers

Located in `molmem_lib.sys_utils` and exported via `molmem_lib`.

```python
from molmem_lib import silenceMolmemLib, isMolmemLibSilenced, MolmemSilence, is_gpu_ready
```

* `silenceMolmemLib(silence=True)`: Globally mutes operational console prints. Can also be used as a context manager:
  ```python
  with silenceMolmemLib():
      sim.updateCrossbarWeights(W)
  ```
* `isMolmemLibSilenced()`: Returns `True` if global silencing is active.
* `is_gpu_ready()`: Checks driver compatibility, compiler readiness, and GPU device availability without throwing exceptions.

---

## 4. Representative Testbench Walkthroughs

The following walkthroughs reproduce canonical testbench patterns used throughout `tb_circuit_level/`, `tb_software_level/`, and `HardwareTestTB/`.

---

### Walkthrough 1: Dynamic SET/RESET Cycling & Thermal Relaxation

Based on `tb_circuit_level/tb_dynamic.py` and `tb_circuit_level/tb_thermal_relaxation.py`. Simulates dynamic switching under pulse trains and monitors self-heating and thermal decay:

```python
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

# 1. Initialize Simulator
sim = MolmemSimulator(instanceName="DynamicDUT")
sim.addCrossbarMatrix(device_type="default", rows=1, cols=1)

t_switch = 2.6432e-3  # 16,520 potentiation pulses at 160 ns
t_stop   = 5.2864e-3  # Followed by 16,520 depression pulses

vpot = PulseSource(node=None, amp=0.9,  period=160e-9, width=80e-9)
vdep = PulseSource(node=None, amp=-0.75, period=160e-9, width=60e-9)

def v_pulse_train(t, sources):
    return sources[0].getVoltage(t) if t < t_switch else sources[1].getVoltage(t)

sim.addBSource(row=1, vFunc=v_pulse_train, sources=[vpot, vdep], edge_points=[t_switch])
sim.addDcSource(col=1, voltage=0.0)

# 2. Run Transient Simulation
sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-4)

# 3. Multi-Panel Waveform Visualization
plotter = MolmemPlotter(sim)
plotter.createFigure("dut_dynamic", title="Single Device Dynamic Switching", subplots=(5, 1), figsize=(10, 10))
plotter.addWaveform("dut_dynamic", 'Voltage',     row=1, subplotIdx=0, color='purple', label='Applied V')
plotter.addWaveform("dut_dynamic", 'Pulses',      row=1, col=1, subplotIdx=1, color='green',  label='State (n)')
plotter.addWaveform("dut_dynamic", 'Conductance', row=1, col=1, subplotIdx=2, color='blue',   label='Conductance (mS)')
plotter.addWaveform("dut_dynamic", 'Current',     row=1, col=1, subplotIdx=3, color='red',    label='Current (A)')
plotter.addWaveform("dut_dynamic", 'Temperature', row=1, col=1, subplotIdx=4, color='orange', label='Junction Temp (K)')

for idx in range(5):
    plotter.formatAxis("dut_dynamic", idx)

plotter.saveFigure("dut_dynamic", "sim_results/tb_dynamic.png")
plotter.showAll()
```

---

### Walkthrough 2: N-by-M Nanoscale Crossbar Transient Simulation

Based on `tb_circuit_level/tb_crossbar_8x8.py` and `tb_circuit_level/tb_crossbar_write.py`. Simulates an $8 \times 8$ crossbar under row pulse trains with sneak-path resolution:

```python
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator(instanceName="Crossbar8x8")
sim.addCrossbarMatrix(device_type="default", rows=8, cols=8)

# Connect pulse sources to select wordlines
v_pulse = PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9)
sim.addVoltageSource(row=1, sourceObj=v_pulse)
sim.addVoltageSource(row=3, sourceObj=v_pulse)

# Ground inactive wordlines and all bitlines
for r in [2, 4, 5, 6, 7, 8]:
    sim.addDcSource(row=r, voltage=0.0)
for c in range(1, 9):
    sim.addDcSource(col=c, voltage=0.0)

sim.run(tEnd=1.6e-6, min_dt=1e-12, tol=1e-3)

# Extract column currents
col_currents = [sim.getCurrent(col=c)[-1] for c in range(1, 9)]
print("Final Column Output Currents:", col_currents)
```

---

### Walkthrough 3: Addressable Tiled Crossbar Array Acceleration

Based on `tb_software_level/tb_tiled_simulator.py` and `tb_software_level/tb_update_tile.py`:

```python
import numpy as np
from molmem_lib import MolmemTiledSimulator

# 1. Instantiate a 64x64 array composed of four 32x32 tiles
tiled_sim = MolmemTiledSimulator(instanceName="MacroAccelerator")
tiled_sim.addTiledCrossbarMatrix(
    total_rows=64, total_cols=64,
    tile_rows=32, tile_cols=32,
    device_type="default",
    weightBits=14, inputBits=14, outputBits=14
)

# 2. Program individual tile (1, 0)
tile_weights = np.random.uniform(0.2, 0.8, (32, 32))
tiled_sim.updateTileCrossbarWeights(tile_r=1, tile_c=0, targetWeightsNormalized=tile_weights, fineTune=False)

# 3. Perform batch inference
x_batch = np.random.uniform(0.0, 1.0, (10, 64)).astype(np.float32)
y_batch = tiled_sim.passTiledCrossbarInput(x_batch, detailedPrint=False)
print("Batch Output Shape:", y_batch.shape)  # (10, 64)
```

---

### Walkthrough 4: Multi-Crossbar Thread-Resource Parallel Programming

Based on `tb_software_level/tb_parallel_crossbars.py`:

```python
import numpy as np
from molmem_lib import MolmemSimulator, updateCrossbarsParallel

# 1. Provision independent differential pair simulators
sim_pos = MolmemSimulator(instanceName="Layer1_Pos")
sim_pos.addCrossbarMatrix(rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14)

sim_neg = MolmemSimulator(instanceName="Layer1_Neg")
sim_neg.addCrossbarMatrix(rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14)

# 2. Generate target weights
W_pos = np.random.uniform(0.1, 0.9, (16, 16))
W_neg = np.random.uniform(0.1, 0.9, (16, 16))

# 3. Dispatch parallel programming across threads / GPU streams
updateCrossbarsParallel(
    simulators=[sim_pos, sim_neg],
    weight_matrices=[W_pos, W_neg],
    detailedPrint=False,
    fineTune=True
)
```

---

### Walkthrough 5: TensorFlow Hardware-in-the-Loop Gradient Training

Based on `tb_software_level/tb_tf_training.py`:

```python
import numpy as np
from molmem_lib import MolmemSimulator, CrossbarDense, EpochTracker, suppress_c_stderr

with suppress_c_stderr():
    import tensorflow as tf

# 1. Instantiate Core Hardware Crossbar
hw_sim = MolmemSimulator(instanceName="TF_Hardware_Core")
hw_sim.addCrossbarMatrix(rows=2, cols=1, weightBits=14, inputBits=14, outputBits=14)

# 2. Construct Keras Model with Physical CrossbarDense Layer
model = tf.keras.Sequential([
    CrossbarDense(units=1, sim_instance=hw_sim, output_scale=1.0),
    tf.keras.layers.Activation('sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Train on Synthetic Dataset
X = np.array([[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
Y = np.array([[1], [0], [1], [0]], dtype=np.float32)

with suppress_c_stderr():
    model.fit(X, Y, epochs=10, batch_size=4, callbacks=[EpochTracker()], verbose=0)
```

---

## 5. Brief Catalog of Auxiliary & Internal API Commands

The following list catalogs all remaining internal, diagnostic, and auxiliary API commands provided by `molmem_lib`:

### Internal Simulation & State Management
* `sim.saveStates()`: Serializes a deep snapshot dictionary of all device physical variables (`n`, `modeState`, `T`, $G$).
* `sim.restoreStates(snapshot)`: Restores device internal variables from a previously saved snapshot dictionary.
* `sim.estimate_gpu_memory_bytes(rows, cols, is_device_basis=False)`: Calculates the theoretical GPU VRAM footprint in bytes for a given crossbar geometry.
* `sim.estimate_cpu_memory_bytes(rows, cols, threads=None, is_device_basis=False)`: Calculates the theoretical CPU RAM footprint in bytes.
* `sim.estimate_gpu_host_ram_bytes(rows, cols, is_device_basis=False)`: Calculates host RAM staging buffer requirements during GPU execution.
* `sim.get_optimal_gpu_slice_size(rows, cols, pad_sources, max_v_pts, is_read=False)`: Computes the maximum safe column/device slice fitting within available GPU VRAM.

### Plotting & Analytics Helpers
* `plotter.wait_all()`: Blocks execution until all background asynchronous image saving threads complete disk writes.

### Hardware Validation & Compiler Utilities
* `clear_gpu_ready_cache()`: Resets cached GPU device accessibility and compatibility flags.
* `get_numba_threading_layer()`: Returns the active low-level threading layer name (`'tbb'`, `'omp'`, or `'workqueue'`).
* `is_interactive_notebook()`: Returns `True` if executing inside a Jupyter, IPython, Colab, or Kaggle notebook environment.
* `is_worker_process()`: Returns `True` if executing within a child multiprocessing worker.
* `get_log_dir()`: Returns the absolute path to `molmem_lib/logs/`.
* `is_logging_enabled()`: Returns `True` if file-based diagnostic run logging is active.
* `cleanup_temp_history_dir()`: Flushes temporary memory-mapped binary history buffers from disk.
* `resolve_and_install_caller_dependencies()`: Inspects calling script imports and automatically resolves missing package dependencies.
* `show_startup_banner()`: Explicitly displays the ASCII terminal hardware summary banner.
* `ensure_startup_banner()`: Displays the startup banner once per process lifecycle unless silenced.
