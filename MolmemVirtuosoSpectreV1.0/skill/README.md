# Cadence Virtuoso CDF and Environment Update Guide

**Subdirectory**: `MolmemVirtuosoSpectreV1.0/skill/`  
**Associated Scripts**: `molmem_cdf.il`, `molmem.scs`, `cds.lib`

---

## 1. Overview

When developing, tuning, or modifying the Molecular Memristor model files in `MolmemVirtuosoSpectreV1.0/models/`, changes may impact:
1. **Model Physics & Parameter Defaults** (`molmem_params.scs`, `molmem_device.scs`, `molmem_noise.scs`, `molmem_core_embedded.va`)
2. **Virtuoso Component Description Format (CDF)** (`molmem_cdf.il`)
3. **Virtuoso Library Definitions** (`cds.lib`)
4. **ADE Simulation States & Netlisting** (Spectre / Maestro)

This guide provides step-by-step instructions for synchronizing Cadence Virtuoso whenever model parameters, terminal definitions, or file paths are modified.

---

## 2. When Do You Need to Update the CDF?

| Type of Modification | Files Affected | Re-run `molmem_cdf.il`? | Action Required |
| :--- | :--- | :---: | :--- |
| **Tuning internal physics equations** | `molmem_core_embedded.va` | **No** | Re-run simulation. Spectre recompiles Verilog-A automatically. |
| **Updating global parameter values** | `molmem_params.scs` | **Optional** | Spectre picks up new values immediately. Update CDF only if you want the schematic symbol property dialog ('Q') to reflect the new defaults. |
| **Adding a new model parameter** | `molmem_device.scs`, `molmem_params.scs` | **Yes** | Add the parameter to `paramList` in `molmem_cdf.il` and reload. |
| **Removing or renaming a parameter** | `molmem_device.scs` | **Yes** | Update `paramList` in `molmem_cdf.il` and reload to remove stale CDF fields. |
| **Changing pin / terminal order** | `molmem_device.scs`, symbol view | **Yes** | Update `termList` and `simInfo->spectre->termOrder` in `molmem_cdf.il`. |
| **Moving repository to a new directory / machine** | `cds.lib`, ADE model path | **No** | Update path entries in `cds.lib` and ADE Model Libraries. |

---

## 3. How to Update the CDF Parameters

### Step 1: Edit `molmem_cdf.il`
Open `MolmemVirtuosoSpectreV1.0/skill/molmem_cdf.il` in your text editor. Locate the `paramList` definition (starting around line 55):

```lisp
paramList = '(
    ; --- Model Selection ---
    ("device_type"            "Device Type (0=Default, 1=Ru_azo)"       "0"           "Model selection: 0=Default, 1=Ru_azo calibrated")

    ; --- Initial State Condition ---
    ("state_init"             "Initial State (0.0=RESET to 1.0=SET)"    "0.0"         "Normalized initial state: 0.0=RESET (n=0), 1.0=SET (n=16520)")

    ; --- Contact & Electrode Physics ---
    ("Rc"                     "Contact Resistance Rc (Ohm)"             "0.008033"    "Lumped electrode series contact resistance")

    ; --- Switching Threshold & Operating Potentials ---
    ("Vth"                    "Switching Threshold Vth (V)"             "0.466409"    "Switching activation threshold voltage")
    ...
)
```

Each entry follows the 4-tuple format:
```lisp
("parameter_name" "Prompt Label in Property Window" "Default_Value_String" "Help Description")
```

- **To modify a default value**: Change the third string (e.g. `"0.008033"` to `"0.010"`).
- **To add a new parameter**: Append a new tuple to `paramList`.
- **To remove a parameter**: Delete its tuple from `paramList`.

### Step 2: Load the Script in Virtuoso CIW
In Cadence Virtuoso, bring up the Command Interpreter Window (CIW) and execute:

```lisp
load("/home/cadence0/Desktop/MolmemVirtuosoSpectreV1.0/skill/molmem_cdf.il")
```
*(Adjust the absolute path to your environment if necessary).*

### Step 3: Re-Register the Base CDF
Run the registration procedure for your destination library:

```lisp
molmemCreateAllCDFs("MolmemLib")
```

This procedure registers the updated CDF across all three supported cell variants:
1. `molmem_device` (primary cell with terminals `Anode`, `Cathode`)
2. `molmem` (universal alias cell with terminals `plus`, `minus`)
3. `MolecularMemristor` (convenience alias)

If you only need to update a single cell:
```lisp
molmemCreateDeviceCDF("MolmemLib" "molmem_device")
```

### What `molmem_cdf.il` Does Behind the Scenes
- Calls `cdfGetBaseCellCDF()` to locate any existing CDF record.
- Calls `cdfDeleteCDF()` to completely wipe stale parameter caches.
- Calls `cdfCreateBaseCellCDF()` to construct a fresh base-level CDF.
- Enables Cadence Expression Language (CEL) parsing (`parseAsCEL="yes"`, `parseAsNumber="yes"`), allowing designers to enter design variables (e.g., `VAR("n0")`) or engineering multipliers (`16.52k`, `80n`, `50m`) directly into symbol properties.
- Binds `simInfo->spectre` netlisting properties:
  - `componentName = cellName` (or subcircuit name)
  - `termOrder = '("Anode" "Cathode")`
  - `netlistProcedure = 'spectreSubckt`

---

## 4. How to Update Cadence Library Setup (`cds.lib`)

If you rename the directory, copy the library to another machine, or change user accounts, update `MolmemVirtuosoSpectreV1.0/cds.lib`:

```text
SOFTINCLUDE /opt/cadence/IC231/share/cdssetup/cds.lib
DEFINE MolmemLib /home/cadence0/Desktop/MolmemVirtuosoSpectreV1.0/MolmemLib
DEFINE SchematicTestbenches /home/cadence0/Desktop/MolmemVirtuosoSpectreV1.0/SchematicTestbenches
```

### Refreshing Virtuoso Library Manager
In Virtuoso CIW, run:
```lisp
ddUpdateLibList()
```
Or in Library Manager: **View -> Refresh**.

---

## 5. How to Update ADE Explorer / Assembler / Maestro

When simulating schematic testbenches via ADE Explorer or Maestro:

1. **Model Library Inclusion**:
   - Open ADE Explorer / Maestro.
   - Go to **Setup -> Model Libraries...**
   - Ensure the path points to the unified top-level model include:
     ```text
     /home/cadence0/Desktop/MolmemVirtuosoSpectreV1.0/models/molmem.scs
     ```
   - Section field: Leave empty (spectre lang).

2. **Force Clean Netlist Regeneration**:
   - If parameter additions or symbol changes do not appear in the simulation netlist:
   - In ADE window: **Simulation -> Netlist -> Recreate**
   - This ensures Spectre compiles a new netlist from the latest CDF definitions rather than reusing a cached `input.scs`.

---

## 6. How to Verify CDF in Virtuoso GUI

To visually inspect that the CDF was registered correctly without opening a schematic:

1. In Virtuoso CIW, go to **Tools -> CDF -> Edit...**
2. In the CDF form:
   - **CDF Type**: Select `Base`
   - **Library Name**: Select `MolmemLib`
   - **Cell Name**: Select `molmem_device` (or `MolecularMemristor`)
3. Check the **Component Parameters** table:
   - Verify all parameters are listed with correct prompts and defaults.
   - Verify `Parse as Number` = `yes` and `Parse as CEL` = `yes`.
4. Switch to the **Simulation Information** tab:
   - Choose Simulator: `spectre`
   - Verify `termOrder` matches the symbol pins (`Anode Cathode`).
   - Verify `componentName` points to the intended Spectre subcircuit.
