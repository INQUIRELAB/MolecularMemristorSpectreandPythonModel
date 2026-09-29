; ==============================================================================
; Cadence Virtuoso CDF (Component Description Format) Definition Script
; File: MolmemVirtuosoSpectreV1.0/skill/molmem_cdf.il
;
; Purpose:
;   Automatically registers complete 2-terminal symbol CDF parameters and Spectre
;   netlisting simulation information for the Molecular Memristor model in
;   Cadence Virtuoso Schematic Editor.
;
; Supported 2-Terminal Cells:
;   1. "molmem_device" : Primary cell with terminals ("Anode" "Cathode")
;   2. "molmem"        : Universal alias cell with terminals ("plus" "minus")
;
; Features:
;   - Exposes all 36 physical, statistical, thermal, and state hyperparameters.
;   - Calibrated default values synchronizing directly with molmem_params.scs.
;   - Full CEL (Cadence Expression Language) and ADE Design Variable support
;     (parseAsCEL="yes", parseAsNumber="yes"), allowing designers to pass raw
;     numbers, scale suffixes (e.g. 16.52k, 10n, 50m), or variables (e.g. VAR("n0")).
;   - Configures simInfo->spectre for seamless ADE / Spectre netlisting.
;
; Usage in Virtuoso CIW (Command Interpreter Window):
;   load("path/to/SpectreModelsV2/skill/molmem_cdf.il")
;   molmemCreateAllCDFs("your_target_library_name")
;   ; Or for a specific cell:
;   molmemCreateDeviceCDF("your_target_library_name" "molmem_device")
; ==============================================================================

procedure( molmemCreateBaseCDF(libName cellName termList)
    let((cellId cdfId paramList)
        cellId = ddGetObj(libName cellName)
        unless(cellId
            warn("[MolMem CDF] Library '%s' or cell '%s' not found. Creating memory cell record...\n" libName cellName)
        )

        ; Retrieve or initialize base cell CDF
        cdfId = cdfGetBaseCellCDF(ddGetObj(libName cellName))
        when(cdfId
            printf("[MolMem CDF] Replacing existing base CDF for %s/%s...\n" libName cellName)
            cdfDeleteCDF(cdfId)
        )
        cdfId = cdfCreateBaseCellCDF(ddGetObj(libName cellName))
        unless(cdfId
            error("[MolMem CDF] Failed to create base CDF for %s/%s. Ensure library is writable." libName cellName)
        )

        ; ----------------------------------------------------------------------
        ; 1. Define Parameter Metadata List
        ; Format: (name prompt defValue units docString)
        ; ----------------------------------------------------------------------
        ; 1. Define Parameter Metadata List:
        ;    Exposes strictly the device type, initial state, and physical parameters
        ;    with explicit physical units (Ohm, V, 1/s, eV, K/W, s, pulses).
        ; ----------------------------------------------------------------------
        paramList = '(
            ; --- Model Selection ---
            ("device_type"            "Device Type (0=Default, 1=Ru_azo)"       "0"           "Model selection: 0=Default, 1=Ru_azo calibrated")

            ; --- Initial State Condition (Normalized 0.0 to 1.0 -> 0 to 16,520) ---
            ("state_init"             "Initial State (0.0=RESET to 1.0=SET)"    "0.0"         "Normalized initial state: 0.0=RESET (n=0), 1.0=SET (n=16520)")

            ; --- Contact & Electrode Physics ---
            ("Rc"                     "Contact Resistance Rc (Ohm)"             "0.008033"    "Lumped electrode series contact resistance")

            ; --- Switching Threshold & Operating Potentials ---
            ("Vth"                    "Switching Threshold Vth (V)"             "0.466409"    "Switching activation threshold voltage")
            ("Vref_pos"               "Positive Reference Vref_pos (V)"         "0.900000"    "Reference positive potentiation pulse voltage")
            ("Vref_neg"               "Negative Reference Vref_neg (V)"         "0.750000"    "Reference negative depotentiation pulse voltage")
            ("V_smooth"               "Voltage Smoothing Window V_smooth (V)"   "0.103776"    "Hyperbolic tangent voltage smoothing window")

            ; --- Transition Kinetics & Spontaneous Relaxation ---
            ("kappa"                  "Base Transition Rate kappa (1/s)"        "1.172865e7"  "Base state transition rate (states/sec)")
            ("K_discharge"            "Thermal Discharge Rate K_discharge (1/s)" "362.88"     "Zero-bias spontaneous thermal discharge rate")

            ; --- Electro-Thermal Junction Physics ---
            ("E_a"                    "Thermal Activation Energy E_a (eV)"      "0.155923"    "Thermal barrier activation energy")
            ("R_th"                   "Thermal Resistance R_th (K/W)"           "52.14"       "Self-heating effective thermal resistance")
            ("tau_th"                 "Thermal Time Constant tau_th (s)"        "1.278e-07"   "Dynamic self-heating relaxation time constant")

            ; --- Empirical Multi-Channel Noise Controls ---
            ("noise_en"                "Noise Enable (0=Off, 1=On)"              "0"           "Master noise engine switch: 0=Off (deterministic), 1=On (active noise)")
            ("read_noise_intensity"    "Temporal Read Noise Scale (0..1)"        "1.0"         "Johnson/flicker read noise scale (ADE Transient Noise with noise_en=1)")
            ("write_noise_intensity"   "Cycle Write Noise Scale (0..1)"          "1.0"         "Cycle-to-cycle Langevin write noise scale (active when noise_en=1)")
            ("d2d_variation_intensity" "D2D Mismatch Scale (0..1)"               "1.0"         "Device-to-device spatial mismatch scale (ADE Monte Carlo mismatch)")
        )

        ; ----------------------------------------------------------------------
        ; 2. Register Each Parameter in the CDF
        ; ----------------------------------------------------------------------
        foreach(item paramList
            let((pName pPrompt pDef)
                pName   = car(item)
                pPrompt = cadr(item)
                pDef    = caddr(item)

                cdfCreateParam( cdfId
                    ?name           pName
                    ?prompt         pPrompt
                    ?defValue       pDef
                    ?type           "string"
                    ?display        "t"
                    ?editable       "t"
                    ?parseAsCEL     "yes"
                    ?parseAsNumber  "yes"
                )
            )
        )

        ; ----------------------------------------------------------------------
        ; 3. Setup Spectre Simulation Information (simInfo->spectre)
        ; ----------------------------------------------------------------------
        cdfId->simInfo = list(nil)
        cdfId->simInfo->spectre = list(nil
            'termOrder          termList
            'termMapping        list(nil 
                                    car(termList)  ":1" 
                                    cadr(termList) ":2"
                                )
            'componentName      cellName
            'netlistProcedure   'ansSpectreSubcktCall
            'otherParameters    '(
                                    device_type state_init Rc
                                    Vth Vref_pos Vref_neg V_smooth
                                    kappa K_discharge
                                    E_a R_th tau_th
                                    noise_en read_noise_intensity write_noise_intensity d2d_variation_intensity
                                 )
        )

        ; Save CDF changes
        cdfSaveCDF(cdfId)
        printf("[MolMem CDF] Successfully registered %d physical & noise CDF parameters for cell '%s' in library '%s'.\n" length(paramList) cellName libName)

        ; ----------------------------------------------------------------------
        ; 4. Ensure 'spectre' Stop-View Exists for OSS Netlister (Fixes OSSHNL-116)
        ; ----------------------------------------------------------------------
        let((symCv)
            symCv = dbOpenCellViewByType(libName cellName "symbol" "" "r")
            when(symCv
                unless(ddGetObj(libName cellName "spectre")
                    errset(
                        dbCopyCellView(symCv libName cellName "spectre")
                        t
                    )
                    if(ddGetObj(libName cellName "spectre") then
                        printf("[MolMem CDF] Automatically created 'spectre' stop-view for cell '%s'.\n" cellName)
                    else
                        printf("[MolMem CDF] Note: To complete OSS netlister stop-view, copy 'symbol' to 'spectre' in Library Manager.\n")
                    )
                )
                dbClose(symCv)
            )
        )
        t
    )
)


; ==============================================================================
; Public Interface Procedures
; ==============================================================================

;; Register 2-terminal primary cell: molmem_device (Anode Cathode)
procedure( molmemCreateDeviceCDF(libName @optional (cellName "molmem_device"))
    molmemCreateBaseCDF(libName cellName '("Anode" "Cathode"))
)

;; Register 2-terminal universal alias cell: molmem (plus minus)
procedure( molmemCreateAliasCDF(libName @optional (cellName "molmem"))
    molmemCreateBaseCDF(libName cellName '("plus" "minus"))
)

;; Convenience: Register all cells at once (molmem_device, molmem, MolecularMemristor)
procedure( molmemCreateAllCDFs(libName)
    let((res1 res2 res3)
        printf("\n==================================================================\n")
        printf("  Registering Molecular Memristor CDF Definitions in: %s\n" libName)
        printf("==================================================================\n")
        res1 = molmemCreateDeviceCDF(libName "molmem_device")
        res2 = molmemCreateAliasCDF(libName "molmem")
        res3 = molmemCreateDeviceCDF(libName "MolecularMemristor")
        printf("------------------------------------------------------------------\n")
        if(res1 && res2 && res3 then
            printf("  [MolMem CDF] All cells (molmem_device, molmem, MolecularMemristor) configured successfully!\n")
        else
            printf("  [MolMem CDF] Configuration completed with warnings (verify cell views).\n")
        )
        printf("==================================================================\n\n")
        t
    )
)

