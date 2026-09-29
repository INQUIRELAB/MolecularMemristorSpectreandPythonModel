import os
import sys

# Limit CPU threading for background libraries to prevent thread over-subscription deadlocks in multiprocessing pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Worker processes constrain Numba to single-threaded mode via numba.set_num_threads(1) in init_worker

import warnings
# Silence SciPy solver deprecation warnings to keep stdout clean for status bar drawing
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from scipy.optimize import differential_evolution, minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy._lib._util import MapWrapper
from scipy.interpolate import PchipInterpolator

# =============================================================================
# Custom Solver with Decaying Mutation (Linear Decay from 1.0 to 0.05)
# =============================================================================
class DecayingDESolver(DifferentialEvolutionSolver):
    def __init__(self, *args, **kwargs):
        self.pool = kwargs.pop('pool', None)
        self.custom_callback = kwargs.pop('callback', None)
        self.current_gen = 0
        self.center_w_base = 0.15
        self.de_bounds = kwargs.get('bounds', None)
        if self.de_bounds is None and len(args) > 1:
            self.de_bounds = args[1]
        super().__init__(*args, callback=None, **kwargs)
        self.callback = self.custom_callback
        self.workers = self.map_with_progress
        self._wrapped_primary_mapper = self.map_with_progress
        self._mapwrapper = MapWrapper(self.map_with_progress)

    def _adapt_curvature_steering(self, nit, improved=True):
        if not hasattr(self, 'x') or self.x is None:
            return
            
        # Re-evaluate best candidate in main process under neutral baseline to cache simulation curves
        cost_function(self.x)
        
        active_err_high = []
        active_err_low = []
        
        global GLOBAL_LAST_G_SAT, GLOBAL_LAST_G_NS, GLOBAL_LAST_G_122
        global GLOBAL_COND_SAT, GLOBAL_COND_NS, GLOBAL_COND_122
        global GLOBAL_NORM_NONLIN_SAT, GLOBAL_NORM_NONLIN_NS, GLOBAL_NORM_NONLIN_122
        global GLOBAL_FIT_SAT, GLOBAL_FIT_NS, GLOBAL_FIT_122
        
        if GLOBAL_FIT_SAT and GLOBAL_LAST_G_SAT is not None and GLOBAL_NORM_NONLIN_SAT is not None:
            res_sat = np.abs(GLOBAL_LAST_G_SAT - GLOBAL_COND_SAT)
            high_mask = GLOBAL_NORM_NONLIN_SAT >= np.median(GLOBAL_NORM_NONLIN_SAT)
            low_mask = ~high_mask
            mean_high = max(float(np.mean(GLOBAL_COND_SAT[high_mask])), 1e-6)
            mean_low = max(float(np.mean(GLOBAL_COND_SAT[low_mask])), 1e-6)
            active_err_high.append(float(np.mean(res_sat[high_mask])) / mean_high)
            active_err_low.append(float(np.mean(res_sat[low_mask])) / mean_low)
            
        if GLOBAL_FIT_NS and GLOBAL_LAST_G_NS is not None and GLOBAL_NORM_NONLIN_NS is not None:
            res_ns = np.abs(GLOBAL_LAST_G_NS - GLOBAL_COND_NS)
            high_mask = GLOBAL_NORM_NONLIN_NS >= np.median(GLOBAL_NORM_NONLIN_NS)
            low_mask = ~high_mask
            mean_high = max(float(np.mean(GLOBAL_COND_NS[high_mask])), 1e-6)
            mean_low = max(float(np.mean(GLOBAL_COND_NS[low_mask])), 1e-6)
            active_err_high.append(float(np.mean(res_ns[high_mask])) / mean_high)
            active_err_low.append(float(np.mean(res_ns[low_mask])) / mean_low)
            
        if GLOBAL_FIT_122 and GLOBAL_LAST_G_122 is not None and GLOBAL_NORM_NONLIN_122 is not None:
            res_122 = np.abs(GLOBAL_LAST_G_122 - GLOBAL_COND_122)
            high_mask = GLOBAL_NORM_NONLIN_122 >= np.median(GLOBAL_NORM_NONLIN_122)
            low_mask = ~high_mask
            mean_high = max(float(np.mean(GLOBAL_COND_122[high_mask])), 1e-6)
            mean_low = max(float(np.mean(GLOBAL_COND_122[low_mask])), 1e-6)
            active_err_high.append(float(np.mean(res_122[high_mask])) / mean_high)
            active_err_low.append(float(np.mean(res_122[low_mask])) / mean_low)
            
        if active_err_high and active_err_low:
            err_h = float(np.mean(active_err_high))
            err_l = float(np.mean(active_err_low))
            old_center = self.center_w_base
            step = 0.02 if improved else 0.01
            if err_h > err_l:
                # High-nonlinearity error dominates -> decrease w_base to sharpen apex/collapse contrast
                self.center_w_base = max(0.0, round(self.center_w_base - step, 3))
                steer_msg = f"Decreased w_base by {step:.2f} (pulling apex & reverse collapse)"
            else:
                # Low-nonlinearity error dominates -> increase w_base to pull body/belly/tail
                self.center_w_base = min(1.0, round(self.center_w_base + step, 3))
                steer_msg = f"Increased w_base by {step:.2f} (pulling body & tails)"
                
            w0 = max(0.0, round(self.center_w_base - 0.05, 3))
            w1 = round(self.center_w_base, 3)
            w2 = min(1.0, round(self.center_w_base + 0.05, 3))
            print_log(
                f"  [Adaptive Steering] Gen {nit}: High-Nonlin Err: {err_h*100:.2f}% vs Low-Nonlin Err: {err_l*100:.2f}%\n"
                f"                      Center w_base: {old_center:.2f} -> {self.center_w_base:.2f} | Cohorts: [{w0:.2f}, {w1:.2f}, {w2:.2f}] | {steer_msg}"
            )

    def map_with_progress(self, func, iterable):
        import time
        global GLOBAL_DASHBOARD_STATUS
        candidates = list(iterable)
        total = len(candidates)
        results = [None] * total
        
        best_in_batch = self.population_energies[0] if len(self.population_energies) > 0 else np.inf
        is_refinement = getattr(self, 'in_refinement', False)
        
        # Partition candidates into 3 cohorts with baseline weights centered around self.center_w_base
        if not is_refinement and total >= 3:
            w0 = max(0.0, round(self.center_w_base - 0.05, 3))
            w1 = round(self.center_w_base, 3)
            w2 = min(1.0, round(self.center_w_base + 0.05, 3))
            n_third = total // 3
            inputs_to_map = []
            for i, cand in enumerate(candidates):
                if i < n_third:
                    w = w0
                elif i < 2 * n_third:
                    w = w1
                else:
                    w = w2
                inputs_to_map.append((cand, w))
        else:
            inputs_to_map = candidates
        
        # Draw the initial 0% progress bar immediately so the dashboard updates to the new generation number
        # and doesn't lag showing the previous generation's status while the first candidate is simulated.
        bar_width = 15
        bar = "░" * bar_width
        f_min, f_max = self.dither if hasattr(self, 'dither') else (0.01, 1.0)
        dither_str = f" | dither: {f_max:.2f} (0.30)"
        
        if hasattr(self, 'x') and self.x is not None:
            x_phys = to_physical_params(self.x)
            kp = x_phys[0]
            al = x_phys[1]
            vTh = x_phys[2]
            E_a = x_phys[8]
            R_th = x_phys[9]
            gamma = x_phys[11]
            p_str = f"kp={kp:.2e} al={al:.2f} vTh={vTh:.2f} Ea={E_a:.2f} Rth={R_th:.1f} g={gamma:.2f}"
        else:
            p_str = "Initializing..."
            
        refine_tag = " [Refinement]" if is_refinement else ""
        best_str = f" | Best Cost: {best_in_batch:.6f}" if not np.isinf(best_in_batch) else " | Best Cost: inf"
        w_base_str = f" | w_base: {self.center_w_base:.2f}"
        
        GLOBAL_DASHBOARD_STATUS = (
            f"Phase 1/2: DE | Gen {self.current_gen}/{self.maxiter}{refine_tag} |{bar}| 0/{total} (Left: {total}){best_str}{dither_str}{w_base_str} | {p_str}"
        )
        import shutil
        cols = shutil.get_terminal_size().columns
        sys.stdout.write("\r\033[K" + GLOBAL_DASHBOARD_STATUS[:cols - 1])
        sys.stdout.flush()
        
        # Use chunksize=1 to enable dynamic load-balancing (work-stealing at the process queue level)
        chunksize = 1
        
        last_update_time = time.time()
        
        for idx, res in enumerate(self.pool.imap(func, inputs_to_map, chunksize=chunksize)):
            results[idx] = res
            if res < best_in_batch:
                best_in_batch = res
                
            current = idx + 1
            remaining = total - current
            
            # Throttle console updates to at most 10 Hz (every 0.1s) to eliminate rendering lag bottlenecks,
            # but guarantee that the first and final frames are drawn.
            now = time.time()
            if current == 1 or current == total or (now - last_update_time) >= 0.1:
                last_update_time = now
                
                percent = int(current / total * 100) if total > 0 else 0
                filled = int(current / total * bar_width) if total > 0 else 0
                bar = "█" * filled + "░" * (bar_width - filled)
                
                f_min, f_max = self.dither if hasattr(self, 'dither') else (0.01, 1.0)
                dither_str = f" | dither: {f_max:.2f} (0.30)"
                
                if hasattr(self, 'population') and self.population is not None and len(self.population) > 0:
                    lo_k, hi_k = self.de_bounds[7]
                    k_pop_phys = 10.0 ** (lo_k + self.population[:, 7] * (hi_k - lo_k))
                    avg_k = float(np.mean(k_pop_phys))
                else:
                    avg_k = None
                
                if hasattr(self, 'x') and self.x is not None:
                    x_phys = to_physical_params(self.x)
                    kp = x_phys[0]
                    al = x_phys[1]
                    vTh = x_phys[2]
                    kDisch = x_phys[7]
                    E_a = x_phys[8]
                    R_th = x_phys[9]
                    gamma = x_phys[11]
                    rLat = x_phys[16] if len(x_phys) >= 17 else 0.5
                    k_str_pop = f" kDisch: avg={avg_k:.2f}|best={kDisch:.2f}" if avg_k is not None else f" kDisch={kDisch:.2f}"
                    p_str = f"kp={kp:.2e} al={al:.2f}{k_str_pop} Ea={E_a:.2f} g={gamma:.2f} rLat={rLat:.2f}"
                else:
                    p_str = "Initializing..."
                
                refine_tag = " [Refinement]" if is_refinement else ""
                best_str = f" | Best Cost: {best_in_batch:.6f}" if not np.isinf(best_in_batch) else " | Best Cost: inf"
                w_base_str = f" | w_base: {self.center_w_base:.2f}"
                
                GLOBAL_DASHBOARD_STATUS = (
                    f"Phase 1/2: DE | Gen {self.current_gen}/{self.maxiter}{refine_tag} |{bar}| {current}/{total} (Left: {remaining}){best_str}{dither_str}{w_base_str} | {p_str}"
                )
                cols = shutil.get_terminal_size().columns
                sys.stdout.write("\r\033[K" + GLOBAL_DASHBOARD_STATUS[:cols - 1])
                sys.stdout.flush()
            
        return results

    def __next__(self):
        if not hasattr(self, 'momentum'):
            self.momentum = np.zeros(self.parameter_count)

        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))
            self._promote_lowest_energy()

        # 10 / 80 / 10 Population Partitioning:
        # - Top 10%: Elites preserved 1:1 without mutation
        # - Middle 80%: Active differential mutation + momentum refinement
        # - Bottom 10%: Elite-anchored semi-uniform exploratory scouts
        n_elites = max(1, int(0.10 * self.num_population_members))
        num_explorers = max(1, int(0.10 * self.num_population_members))

        # 1. Calculate centroid of the current top 10% elites BEFORE sorting/updates
        sorted_indices_prev = np.argsort(self.population_energies)
        centroid_prev = np.mean(self.population[sorted_indices_prev[:n_elites]], axis=0)

        # 2. Sort the entire population by energy (cost)
        sorted_indices = np.argsort(self.population_energies)
        self.population = self.population[sorted_indices]
        self.population_energies = self.population_energies[sorted_indices]
        self.feasible = self.feasible[sorted_indices]
        self.constraint_violation = self.constraint_violation[sorted_indices]

        # Apply mutation scale dither if active
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(self.dither[0], self.dither[1])

        # 3. Perform mutation/recombination on the active 90% (indices n_elites to N-1)
        if self._nfev >= self.maxfun:
            raise StopIteration

        active_indices = np.arange(n_elites, self.num_population_members)
        trial_pop_active = self._mutate_many(active_indices)

        # 4. Elite-Anchored Semi-Uniform Exploration (Bottom 10% of total population)
        # Replaces the lowest tier of trial candidates: 50% traits copied from best candidate,
        # 50% traits sampled uniformly across full bounds to maintain radical global diversity
        n_random_traits = self.parameter_count // 2  # 9 of 19 parameters
        
        for i in range(len(trial_pop_active) - num_explorers, len(trial_pop_active)):
            explorer = np.copy(self.population[0])  # Copy best candidate's traits (normalized [0, 1])
            rand_traits = self.random_number_generator.choice(
                self.parameter_count, size=n_random_traits, replace=False
            )
            explorer[rand_traits] = self.random_number_generator.uniform(
                0.0, 1.0, size=n_random_traits
            )
            trial_pop_active[i] = explorer

        # 5. Apply momentum push randomly (50% probability) to standard mutated candidates (excluding explorers)
        for i in range(len(trial_pop_active) - num_explorers):
            if self.random_number_generator.uniform(0, 1) < 0.5:
                # Random scale between 1% and 50% of momentum vector
                scale = self.random_number_generator.uniform(0.01, 0.50)
                trial_pop_active[i] += scale * self.momentum

        # 6. Spatial De-Clustering & Historical Candidate Memory Check
        # Prevents redundant evaluations by enforcing a minimum spatial exclusion radius
        # around the top elites and historical champions in normalized parameter space
        if not hasattr(self, 'historical_archive'):
            self.historical_archive = []
            
        # Keep rolling archive of the top elites across generations (up to 300 unique coordinates)
        top_elite_batch = self.population[:min(50, n_elites)]
        for elite_cand in top_elite_batch:
            if len(self.historical_archive) == 0 or np.min(np.linalg.norm(self.historical_archive - elite_cand, axis=1)) > 0.005:
                self.historical_archive.append(np.copy(elite_cand))
                if len(self.historical_archive) > 300:
                    self.historical_archive.pop(0)
                    
        archive_mat = np.array(self.historical_archive)
        MIN_CLUSTER_DIST = 0.008  # Minimum Euclidean distance in normalized 19D space (~0.2% per dim)
        
        n_declustered = 0
        n_to_check = len(trial_pop_active) - num_explorers
        if len(archive_mat) > 0 and n_to_check > 0:
            # Compute distances of all trials to all archived elites
            diffs = trial_pop_active[:n_to_check, None, :] - archive_mat[None, :, :]
            dist_sq = np.sum(diffs ** 2, axis=2)
            min_dist_idx = np.argmin(dist_sq, axis=1)
            min_dists = np.sqrt(dist_sq[np.arange(n_to_check), min_dist_idx])
            
            crowded_mask = min_dists < MIN_CLUSTER_DIST
            crowded_indices = np.where(crowded_mask)[0]
            n_declustered = len(crowded_indices)
            
            for idx in crowded_indices:
                # Nudge crowded trial radially outward along random unit vector
                rand_vec = self.random_number_generator.standard_normal(self.parameter_count)
                norm_v = np.linalg.norm(rand_vec)
                rand_vec /= norm_v if norm_v > 1e-12 else 1.0
                push = MIN_CLUSTER_DIST + self.random_number_generator.uniform(0.005, 0.025)
                trial_pop_active[idx] = trial_pop_active[idx] + push * rand_vec

        self.last_n_declustered = n_declustered

        # Enforce bounds
        self._ensure_constraint(trial_pop_active)

        # Calculate feasibility and energies for the active trials
        feasible_active, cv_active = self._calculate_population_feasibilities(trial_pop_active)
        trial_energies_active = np.full(len(active_indices), np.inf)

        trial_energies_active[feasible_active] = self._calculate_population_energies(
            trial_pop_active[feasible_active])

        # Compare trials against active parent candidates (pairwise selection)
        loc = [self._accept_trial(*val) for val in
               zip(trial_energies_active, feasible_active, cv_active,
                   self.population_energies[n_elites:], self.feasible[n_elites:], self.constraint_violation[n_elites:])]
        loc = np.array(loc)

        # Update active portion with the accepted trials
        self.population[n_elites:] = np.where(loc[:, np.newaxis], trial_pop_active, self.population[n_elites:])
        self.population_energies[n_elites:] = np.where(loc, trial_energies_active, self.population_energies[n_elites:])
        self.feasible[n_elites:] = np.where(loc, feasible_active, self.feasible[n_elites:])
        self.constraint_violation[n_elites:] = np.where(loc[:, np.newaxis], cv_active, self.constraint_violation[n_elites:])

        # Put the lowest energy into index 0 (best candidate promotion)
        self._promote_lowest_energy()

        # 6. Calculate centroid of the new top 10% elites AFTER updates
        sorted_indices_curr = np.argsort(self.population_energies)
        centroid_curr = np.mean(self.population[sorted_indices_curr[:n_elites]], axis=0)

        # 7. Update running momentum vector based on elite centroid movement
        step_diff = centroid_curr - centroid_prev
        self.momentum = 0.7 * self.momentum + 0.3 * step_diff

        return self.x, self.population_energies[0]

    def solve(self):
        if np.all(np.isinf(self.population_energies)):
            # Draw initial dashboard to let the user know we are evaluating Generation 0 / Initialization
            update_dashboard(phase=1, current=0, total=self.maxiter, cost=np.inf, best_xk=self.x, dither=[0.01, 1.0], w_base=self.center_w_base)
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(self.population[self.feasible]))
            self._promote_lowest_energy()
            if getattr(self, 'callback', None) is not None:
                try:
                    self.callback(self.x, nit=0)
                except TypeError:
                    self.callback(self.x)
                except Exception:
                    import traceback
                    traceback.print_exc()

        nit = 0
        status_message = "Optimization terminated successfully."
        warning_flag = False

        best_cost = np.inf
        no_improvement_count = 0
        decay_nit = 1

        for nit in range(1, self.maxiter + 1):
            self.current_gen = nit
            # Linearly decay max dither mutation bound from 1.0 down to 0.05
            if self.maxiter > 1:
                fraction = (decay_nit - 1) / (self.maxiter - 1)
            else:
                fraction = 0.0
            
            f_min = 0.01
            f_max = max(0.30, 1.0 - (1.0 - 0.05) * fraction)
            self.dither = [f_min, f_max]
            
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                break
                
            current_cost = self.population_energies[0]
            improved = False
            if current_cost < best_cost - 1e-5:
                best_cost = current_cost
                no_improvement_count = 0
                improved = True
                
                if REFINEMENT_SWEEP_STEPS > 0:
                    print_log(f"  [DE] Generation {nit}/{self.maxiter} | New Best Cost: {current_cost:.6f} | Running coordinate refinement sweep ({self.parameter_count * REFINEMENT_SWEEP_STEPS} points)...")
                    
                    # Unnormalize best candidate to optimizer space
                    x_best_opt = np.zeros(self.parameter_count)
                    for col_idx in range(self.parameter_count):
                        lo, hi = self.de_bounds[col_idx]
                        x_best_opt[col_idx] = lo + self.population[0, col_idx] * (hi - lo)
                        
                    # Generate coordinate-wise sweep grid
                    total_refine_samples = self.parameter_count * REFINEMENT_SWEEP_STEPS
                    x_refine_samples = np.zeros((total_refine_samples, self.parameter_count))
                    for i in range(total_refine_samples):
                        x_refine_samples[i, :] = x_best_opt.copy()
                        
                    grids = []
                    for col_idx in range(self.parameter_count):
                        lo, hi = self.de_bounds[col_idx]
                        grid = np.linspace(lo, hi, REFINEMENT_SWEEP_STEPS)
                        grids.append(grid)
                        start_row = col_idx * REFINEMENT_SWEEP_STEPS
                        x_refine_samples[start_row : start_row + REFINEMENT_SWEEP_STEPS, col_idx] = grid
                        
                    # Evaluate grid in parallel using the solver's pool workers
                    self.in_refinement = True
                    refine_costs = self.workers(self.func, x_refine_samples)
                    self.in_refinement = False
                    refine_costs = np.array(refine_costs)
                    
                    # Check if any refinement point is strictly better
                    best_refine_idx = np.argmin(refine_costs)
                    cost_refine = refine_costs[best_refine_idx]
                    
                    if cost_refine < current_cost - 1e-5:
                        x_refine_best = x_refine_samples[best_refine_idx]
                        # Map back to normalized space [0, 1]
                        u_refine_best = np.zeros(self.parameter_count)
                        for col_idx in range(self.parameter_count):
                            lo, hi = self.de_bounds[col_idx]
                            u_refine_best[col_idx] = (x_refine_best[col_idx] - lo) / (hi - lo)
                            
                        # Overwrite best member and energy
                        self.population[0] = u_refine_best
                        self.population_energies[0] = cost_refine
                        
                        # Re-sort to maintain order
                        self._promote_lowest_energy()
                        
                        print_log(f"  [Refinement] Successfully polished cost from {current_cost:.6f} to {cost_refine:.6f}!")
                        current_cost = cost_refine
                        best_cost = cost_refine
                    else:
                        print_log(f"  [Refinement] No improvement found (retained cost: {current_cost:.6f}).")
            else:
                no_improvement_count += 1

            # Calculate population kDischarge statistics
            lo_k, hi_k = self.de_bounds[7]
            k_pop_phys = 10.0 ** (lo_k + self.population[:, 7] * (hi_k - lo_k))
            avg_k = float(np.mean(k_pop_phys))
            med_k = float(np.median(k_pop_phys))
            best_k = float(to_physical_params(self.x)[7]) if hasattr(self, 'x') and self.x is not None else 0.0

            # Print status update including best cost, w_base, kDischarge, de-clustering count, and momentum norm
            mom_norm = np.linalg.norm(self.momentum) if hasattr(self, 'momentum') else 0.0
            n_decluster = getattr(self, 'last_n_declustered', 0)
            print_log(f"[DE] Generation {nit}/{self.maxiter} | Best Cost: {current_cost:.6f} | w_base: {self.center_w_base:.2f} | kDischarge: avg={avg_k:.2f} (med={med_k:.2f}, best={best_k:.2f}) | de-clustered: {n_decluster} | dither: [{f_min:.4f}, {f_max:.4f} (0.30)] | momentum norm: {mom_norm:.4f}")
            update_dashboard(phase=1, current=nit, total=self.maxiter, cost=current_cost, best_xk=self.x, dither=self.dither, w_base=self.center_w_base, avg_k=avg_k)
            
            # Adaptive Nonlinearity Error Steering (every generation starting after generation 2 concludes)
            if nit >= 2 and hasattr(self, 'x') and self.x is not None:
                try:
                    self._adapt_curvature_steering(nit, improved=improved)
                except Exception:
                    pass
            
            # Incremental save callback
            if getattr(self, 'callback', None) is not None:
                try:
                    self.callback(self.x, nit=nit)
                except TypeError:
                    self.callback(self.x)
                except Exception:
                    import traceback
                    traceback.print_exc()
                
            if improved:
                decay_nit = min(self.maxiter, decay_nit + 1)
            else:
                if f_max > 0.30:
                    jump_steps = no_improvement_count
                    decay_nit = min(self.maxiter, decay_nit + jump_steps)
                    print_log(f"  [DE] Stagnation detected ({no_improvement_count} gens). Jumping decay ahead by {jump_steps} iterations to reduce dither.")
                else:
                    if no_improvement_count >= 3:
                        print_log(f"  [DE] Early stopping triggered: no cost improvement in {no_improvement_count} generations at the dither floor (0.30).")
                        break
                    decay_nit = min(self.maxiter, decay_nit + 1)

            if self.converged():
                break
        else:
            warning_flag = True

        return self._result(nit=nit, message=status_message, warning_flag=warning_flag)

def get_cost_for_xk(xk):
    global GLOBAL_LAST_X, GLOBAL_LAST_COST
    if GLOBAL_LAST_X is not None and np.allclose(xk, GLOBAL_LAST_X, rtol=1e-5, atol=1e-5):
        return GLOBAL_LAST_COST
    return cost_function(xk)

class LocalMinimizationTracker:
    def __init__(self, start_cost, best_overall_cost_ref, on_new_best_callback, start_x, basin_idx, total_basins):
        self.best_cost = start_cost
        self.best_x = np.copy(start_x)
        self.no_improvement_count = 0
        self.best_overall_cost_ref = best_overall_cost_ref
        self.on_new_best_callback = on_new_best_callback
        self.basin_idx = basin_idx
        self.total_basins = total_basins

    def __call__(self, xk):
        cost = get_cost_for_xk(xk)
        update_dashboard(phase=2, current=self.basin_idx, total=self.total_basins, cost=self.best_overall_cost_ref[0], best_xk=xk)
        
        improvement = self.best_cost - cost
        if cost < self.best_cost - 1e-5:
            # Check if this improvement is substantial compared to the gap to best overall cost
            is_substantial = True
            if cost > self.best_overall_cost_ref[0]:
                gap = cost - self.best_overall_cost_ref[0]
                min_required = max(1e-3, 0.05 * gap)
                if improvement < min_required:
                    is_substantial = False
                    
            self.best_cost = cost
            self.best_x = np.copy(xk)
            
            if is_substantial:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                print_log(f"  [L-BFGS-B] Basin {self.basin_idx}/{self.total_basins} | Cost: {cost:.6f} (Stagnant: improvement {improvement:.6f} < required {min_required:.6f}) | Count: {self.no_improvement_count}/3")
            
            if cost < self.best_overall_cost_ref[0]:
                self.best_overall_cost_ref[0] = cost
                self.on_new_best_callback(xk)
        else:
            self.no_improvement_count += 1
            print_log(f"  [L-BFGS-B] Basin {self.basin_idx}/{self.total_basins} | Iteration cost: {cost:.6f} (No improvement count: {self.no_improvement_count}/3)")
            
        if self.no_improvement_count >= 3:
            print_log(f"  [L-BFGS-B] Early stopping triggered: stagnant convergence in basin.")
            raise StopIteration("Stagnant convergence in basin")



# Add PythonSimulator to sys.path so we can import molmem_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PythonSimulator')))

import struct
import hashlib
import subprocess
import glob

def check_core_bundle_alignment(bundle_path, lib_dir):
    """
    Checks if molmem_core.bin exists and its attestation seal validates
    against the current lib_dir footprint. Returns True if aligned, False otherwise.
    """
    if not os.path.isfile(bundle_path):
        return False
    try:
        with open(bundle_path, "rb") as f:
            raw_bytes = f.read(48)
        if len(raw_bytes) < 48:
            return False
        magic = raw_bytes[:8]
        if magic != b"MOLMEM\x03\x00":
            return False
        magic, version, num_sections, salt, wrapped_dek, auth_token, reserved = struct.unpack(
            "<8sIIQQQQ", raw_bytes[:48]
        )
        if version != 3:
            return False
            
        hasher = hashlib.sha256()
        source_files = sorted(glob.glob(os.path.join(lib_dir, "*.py")))
        for py_file in source_files:
            if os.path.basename(py_file) == "devices.py":
                continue
            try:
                with open(py_file, 'rb') as f:
                    content = f.read().replace(b'\r\n', b'\n')
                    hasher.update(content)
            except Exception:
                pass
        footprint = hasher.hexdigest()
        
        digest = hashlib.sha256(footprint.encode('ascii')).digest()
        kek = struct.unpack("<Q", digest[:8])[0]
        
        mask64 = 0xFFFFFFFFFFFFFFFF
        salt_scramble = (salt * 6364136223846793005 + 1442695040888963407) & mask64
        dek = (wrapped_dek ^ kek ^ salt_scramble) & mask64
        
        expected_token = struct.unpack("<Q", hashlib.sha256(struct.pack("<Q", dek)).digest()[:8])[0]
        if expected_token == auth_token:
            return True

        # Canonical footprint fallback (aligns with sys_utils.py)
        CANONICAL_FOOTPRINT = "2ab6592f9b19e9c23e46c7350c95ab6a2c01f117ce3bc3d4f30ef8ea1337050d"
        kek_digest_canon = hashlib.sha256(CANONICAL_FOOTPRINT.encode('ascii')).digest()
        kek_canon = struct.unpack("<Q", kek_digest_canon[:8])[0]
        dek_canon = (wrapped_dek ^ kek_canon ^ salt_scramble) & mask64
        token_canon = struct.unpack("<Q", hashlib.sha256(struct.pack("<Q", dek_canon)).digest()[:8])[0]
        return token_canon == auth_token
    except Exception:
        return False

def ensure_molmem_core_sealed(molmem_lib_dir=None, force=False):
    """
    Ensures that molmem_core.bin in molmem_lib/data is sealed with the exact
    codebase integrity attestation key matching the current molmem_lib source.
    """
    if multiprocessing.current_process().name != 'MainProcess':
        return True

    if molmem_lib_dir is None:
        try:
            import molmem_lib
            molmem_lib_dir = os.path.dirname(os.path.abspath(molmem_lib.__file__))
        except Exception:
            molmem_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PythonSimulator", "molmem_lib")
            
    if not os.path.exists(molmem_lib_dir):
        print(f"[Attestation Error] Could not find molmem_lib directory at: {molmem_lib_dir}", flush=True)
        return False
        
    data_dir = os.path.join(molmem_lib_dir, "data")
    bundle_path = os.path.join(data_dir, "molmem_core.bin")
    
    if not force and check_core_bundle_alignment(bundle_path, molmem_lib_dir):
        return True
        
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compiler_dir = os.path.join(workspace_root, "DevicePhysicsCompiler")
    compiler_script = os.path.join(compiler_dir, "compile_physics_data.py")
    
    if not os.path.isfile(compiler_script):
        if not os.path.isfile(bundle_path):
            print(
                "\n"
                "========================================================================================\n"
                " [!] MOLMEM INITIALIZATION ERROR: Physics core bundle 'molmem_core.bin' was not found\n"
                "     in 'PythonSimulator/molmem_lib/data', and 'DevicePhysicsCompiler' is not available\n"
                "     in this distribution to compile it.\n"
                "     Please provide a valid 'molmem_core.bin' asset file.\n"
                "========================================================================================\n",
                flush=True
            )
            return False
        else:
            print(
                "\n"
                "========================================================================================\n"
                " [!] MOLMEM INTEGRITY WARNING: 'molmem_core.bin' failed cryptographic attestation\n"
                "     against the current 'molmem_lib' source files, and 'DevicePhysicsCompiler' is not\n"
                "     present in this environment to re-seal it.\n"
                "     Please restore the original unmodified 'molmem_lib' files or obtain an updated bundle.\n"
                "========================================================================================\n",
                flush=True
            )
            return False
        
    print("[Attestation] Re-sealing molmem_core.bin to align with current molmem_lib codebase...", flush=True)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("compile_physics_data", compiler_script)
        if spec and spec.loader:
            compiler_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compiler_mod)
            success = compiler_mod.compile_all(source_dir=compiler_dir, target_dir=data_dir)
            if success:
                print("[Attestation] molmem_core.bin successfully re-sealed and verified bit-exact.", flush=True)
                return True
    except Exception as e:
        print(f"[Attestation Warning] In-process compilation error: {e}. Falling back to subprocess...", flush=True)
        
    try:
        res = subprocess.run([sys.executable, compiler_script], cwd=compiler_dir, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print("[Attestation] molmem_core.bin successfully re-sealed via subprocess.", flush=True)
            return True
        else:
            print(f"[Attestation Error] Compiler exited with code {res.returncode}: {res.stderr[:200]}", flush=True)
            return False
    except Exception as e:
        print(f"[Attestation Error] Could not run compiler: {e}", flush=True)
        return False

from molmem_lib import MolmemSimulator
from molmem_lib.sources import PulseSource
from molmem_lib.devices import default_device

def get_default_ranges(data_dir=None):
    t_period = 160e-9
    vpot = PulseSource(node=None, amp=0.9, period=t_period, width=80e-9)
    vdep = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    
    # 1. Non-saturation
    sim_ns = MolmemSimulator(device='cpu', backend='numba')
    sim_ns.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    t_stop_ns = 33040.0 * t_period
    t_switch_ns = t_stop_ns / 2.0
    def v_pulse_train_ns(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
    sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=[t_switch_ns])
    sim_ns.addDcSource(col=1, voltage=0.0)
    sim_ns.run(tEnd=t_stop_ns, min_dt=1e-12, tol=1e-3, recordHistory=['g', 'state', 'f22'])
    g_ns = sim_ns.getConductance(row=1, col=1) * 1e3
    
    # 2. Saturation
    sim_sat = MolmemSimulator(device='cpu', backend='numba')
    sim_sat.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    t_stop_sat = 52000.0 * t_period
    t_switch_sat = t_stop_sat / 2.0
    def v_pulse_train_sat(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
    sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=[t_switch_sat])
    sim_sat.addDcSource(col=1, voltage=0.0)
    sim_sat.run(tEnd=t_stop_sat, min_dt=1e-12, tol=1e-3, recordHistory=['g', 'state', 'f22'])
    g_sat = sim_sat.getConductance(row=1, col=1) * 1e3
    
    # 3. 1.22V Saturation
    sim_122 = MolmemSimulator(device='cpu', backend='numba')
    sim_122.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    t_stop_122 = 33040.0 * t_period
    t_switch_122 = t_stop_122 / 2.0
    vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
    def v_pulse_train_122(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
    sim_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep], edge_points=[t_switch_122])
    sim_122.addDcSource(col=1, voltage=0.0)
    sim_122.run(tEnd=t_stop_122, min_dt=1e-12, tol=1e-3, recordHistory=['g', 'state', 'f22'])
    g_122 = sim_122.getConductance(row=1, col=1) * 1e3
    
    return (np.min(g_ns), np.max(g_ns)), (np.min(g_sat), np.max(g_sat)), (np.min(g_122), np.max(g_122))

# =============================================================================
# Global variables for cost function evaluation to avoid IPC serialization overhead
# =============================================================================
GLOBAL_COND_NS = None
GLOBAL_PULSES_NS = None
GLOBAL_COND_SAT = None
GLOBAL_PULSES_SAT = None
GLOBAL_COND_122 = None
GLOBAL_PULSES_122 = None
GLOBAL_BASE_PARAMS = None
GLOBAL_T_PERIOD = None
GLOBAL_T_WIDTH = None
GLOBAL_VPOT = None
GLOBAL_VDEP = None

GLOBAL_SIM_NS = None
GLOBAL_SIM_SAT = None
GLOBAL_SIM_122 = None
GLOBAL_VPOT_122 = None
GLOBAL_VDEP_122 = None

GLOBAL_DIFF_NS = None
GLOBAL_DIFF_SAT = None
GLOBAL_DIFF_122 = None

GLOBAL_LOG_NS = None
GLOBAL_LOG_SAT = None
GLOBAL_LOG_122 = None

GLOBAL_VAR_NS = 1.0
GLOBAL_VAR_DIFF_NS = 1.0
GLOBAL_VAR_LOG_NS = 1.0

GLOBAL_VAR_SAT = 1.0
GLOBAL_VAR_DIFF_SAT = 1.0
GLOBAL_VAR_LOG_SAT = 1.0

GLOBAL_VAR_122 = 1.0
GLOBAL_VAR_DIFF_122 = 1.0
GLOBAL_VAR_LOG_122 = 1.0

GLOBAL_SIM_TOL_VAL = None
GLOBAL_BEST_COST_VAL = None
GLOBAL_LAST_X = None
GLOBAL_LAST_COST = None

GLOBAL_FIT_NS = True
GLOBAL_FIT_SAT = True
GLOBAL_FIT_122 = True

GLOBAL_WEIGHTS_NS = None
GLOBAL_WEIGHTS_SAT = None
GLOBAL_WEIGHTS_122 = None
GLOBAL_SUM_WEIGHTS_NS = 1.0
GLOBAL_SUM_WEIGHTS_SAT = 1.0
GLOBAL_SUM_WEIGHTS_122 = 1.0
GLOBAL_NORM_NONLIN_NS = None
GLOBAL_NORM_NONLIN_SAT = None
GLOBAL_NORM_NONLIN_122 = None
GLOBAL_LAST_G_NS = None
GLOBAL_LAST_G_SAT = None
GLOBAL_LAST_G_122 = None

def compute_normalized_nonlinearity(x_vals, y_vals, raw_x=None, raw_y=None, window_fraction=0.04):
    """
    Computes normalized second-order curvature / chord deviation (0.0 to 1.0).
    If raw sparse empirical coordinates are provided, curvature is computed on the raw
    points and interpolated onto the dense curve to ensure the full S-curve knees, shoulders,
    and apex receive strong gradient emphasis without being suppressed by dense grid spacing.
    If only dense points are provided, evaluates chord deviation across a physical window
    (default 4% of curve width).
    """
    if raw_x is not None and raw_y is not None and len(raw_x) < len(x_vals) and len(raw_x) > 2:
        s_idx = np.argsort(raw_x)
        rx = np.asarray(raw_x, dtype=np.float64)[s_idx]
        ry = np.asarray(raw_y, dtype=np.float64)[s_idx]
        raw_nonlin = compute_normalized_nonlinearity(rx, ry)
        return np.interp(x_vals, rx, raw_nonlin)

    n = len(y_vals)
    if n <= 2:
        return np.zeros(n, dtype=np.float64)
    
    step = max(1, int(n * window_fraction / 2.0))
    d_nonlin = np.zeros(n, dtype=np.float64)
    for i in range(n):
        i0 = max(0, i - step)
        i2 = min(n - 1, i + step)
        if i2 - i0 < 2:
            continue
        x0, x1, x2 = float(x_vals[i0]), float(x_vals[i]), float(x_vals[i2])
        y0, y1, y2 = float(y_vals[i0]), float(y_vals[i]), float(y_vals[i2])
        dx = x2 - x0
        if abs(dx) > 1e-12:
            y_chord = y0 + (y2 - y0) * ((x1 - x0) / dx)
            d_nonlin[i] = abs(y1 - y_chord)
        else:
            d_nonlin[i] = 0.0
            
    max_d = np.max(d_nonlin)
    if max_d > 1e-12:
        return d_nonlin / max_d
    return np.zeros(n, dtype=np.float64)

def compute_nonlinearity_weights(x_vals, y_vals, w_base=0.50, raw_x=None, raw_y=None):
    norm_d = compute_normalized_nonlinearity(x_vals, y_vals, raw_x=raw_x, raw_y=raw_y)
    return w_base + (1.0 - w_base) * norm_d

def create_pchip_dense_dataset(pulses, cond, num_points=1000):
    """
    Interpolate empirical data points using monotonic PCHIP (matching compare_fit.py)
    and resample into exactly num_points uniformly spaced points.
    """
    p_vals = np.asarray(pulses, dtype=np.float64)
    c_vals = np.asarray(cond, dtype=np.float64)
    
    sort_idx = np.argsort(p_vals)
    p_sorted = p_vals[sort_idx]
    c_sorted = c_vals[sort_idx]
    
    u_idx = [0]
    for i in range(1, len(p_sorted)):
        if p_sorted[i] > p_sorted[u_idx[-1]] + 1e-5:
            u_idx.append(i)
            
    p_unique = p_sorted[u_idx]
    c_unique = c_sorted[u_idx]
    
    pchip = PchipInterpolator(p_unique, c_unique)
    p_dense = np.linspace(p_unique[0], p_unique[-1], num_points)
    c_dense = pchip(p_dense)
    return p_dense, c_dense

def init_worker(cond_ns, pulses_ns, cond_sat, pulses_sat, cond_122, pulses_122, 
                base_params, t_period, t_width, vpot, vdep, tol_val=None, best_cost_val=None,
                fit_ns=True, fit_sat=True, fit_122=True):
    import signal, os, multiprocessing
    if multiprocessing.current_process().name != 'MainProcess':
        signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))
        os.environ["MOLMEM_ACTIVE_WORKER_TASK"] = "1"

    # Force sub-libraries to run strictly single-threaded inside worker processes
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    # Capture compact raw empirical points before expanding into dense 1,000-point monotonic PCHIP curve
    raw_p_ns, raw_c_ns = (pulses_ns, cond_ns) if (cond_ns is not None and len(cond_ns) != 1000) else (None, None)
    raw_p_sat, raw_c_sat = (pulses_sat, cond_sat) if (cond_sat is not None and len(cond_sat) != 1000) else (None, None)
    raw_p_122, raw_c_122 = (pulses_122, cond_122) if (cond_122 is not None and len(cond_122) != 1000) else (None, None)

    # Expand compact raw empirical points into dense 1,000-point monotonic PCHIP curve in worker memory
    if fit_ns and cond_ns is not None and len(cond_ns) != 1000:
        pulses_ns, cond_ns = create_pchip_dense_dataset(pulses_ns, cond_ns, num_points=1000)
    if fit_sat and cond_sat is not None and len(cond_sat) != 1000:
        pulses_sat, cond_sat = create_pchip_dense_dataset(pulses_sat, cond_sat, num_points=1000)
    if fit_122 and cond_122 is not None and len(cond_122) != 1000:
        pulses_122, cond_122 = create_pchip_dense_dataset(pulses_122, cond_122, num_points=1000)

    global GLOBAL_COND_NS, GLOBAL_PULSES_NS, GLOBAL_COND_SAT, GLOBAL_PULSES_SAT
    global GLOBAL_COND_122, GLOBAL_PULSES_122, GLOBAL_BASE_PARAMS
    global GLOBAL_T_PERIOD, GLOBAL_T_WIDTH, GLOBAL_VPOT, GLOBAL_VDEP
    global GLOBAL_SIM_NS, GLOBAL_SIM_SAT, GLOBAL_SIM_122
    global GLOBAL_VPOT_122, GLOBAL_VDEP_122
    global GLOBAL_DIFF_NS, GLOBAL_DIFF_SAT, GLOBAL_DIFF_122
    global GLOBAL_LOG_NS, GLOBAL_LOG_SAT, GLOBAL_LOG_122
    global GLOBAL_SIM_TOL_VAL, GLOBAL_BEST_COST_VAL
    global GLOBAL_VAR_NS, GLOBAL_VAR_DIFF_NS, GLOBAL_VAR_LOG_NS
    global GLOBAL_VAR_SAT, GLOBAL_VAR_DIFF_SAT, GLOBAL_VAR_LOG_SAT
    global GLOBAL_VAR_122, GLOBAL_VAR_DIFF_122, GLOBAL_VAR_LOG_122
    global GLOBAL_FIT_NS, GLOBAL_FIT_SAT, GLOBAL_FIT_122
    global GLOBAL_WEIGHTS_NS, GLOBAL_WEIGHTS_SAT, GLOBAL_WEIGHTS_122
    global GLOBAL_SUM_WEIGHTS_NS, GLOBAL_SUM_WEIGHTS_SAT, GLOBAL_SUM_WEIGHTS_122
    global GLOBAL_NORM_NONLIN_NS, GLOBAL_NORM_NONLIN_SAT, GLOBAL_NORM_NONLIN_122
    
    GLOBAL_COND_NS = cond_ns
    GLOBAL_PULSES_NS = pulses_ns
    GLOBAL_COND_SAT = cond_sat
    GLOBAL_PULSES_SAT = pulses_sat
    GLOBAL_COND_122 = cond_122
    GLOBAL_PULSES_122 = pulses_122
    GLOBAL_BASE_PARAMS = base_params
    GLOBAL_T_PERIOD = t_period
    GLOBAL_T_WIDTH = t_width
    GLOBAL_VPOT = vpot
    GLOBAL_VDEP = vdep
    GLOBAL_SIM_TOL_VAL = tol_val
    GLOBAL_BEST_COST_VAL = best_cost_val
    GLOBAL_FIT_NS = fit_ns
    GLOBAL_FIT_SAT = fit_sat
    GLOBAL_FIT_122 = fit_122

    if fit_ns and cond_ns is not None:
        GLOBAL_DIFF_NS = np.diff(cond_ns)
        GLOBAL_LOG_NS = np.log10(np.clip(cond_ns, 5e-5, None))
        GLOBAL_VAR_NS = max(np.var(cond_ns), 1e-6)
        GLOBAL_VAR_DIFF_NS = max(np.var(GLOBAL_DIFF_NS), 1e-9)
        GLOBAL_VAR_LOG_NS = max(np.var(GLOBAL_LOG_NS), 1e-6)
        GLOBAL_NORM_NONLIN_NS = compute_normalized_nonlinearity(pulses_ns, cond_ns, raw_x=raw_p_ns, raw_y=raw_c_ns)
        GLOBAL_WEIGHTS_NS = compute_nonlinearity_weights(pulses_ns, cond_ns, w_base=0.50, raw_x=raw_p_ns, raw_y=raw_c_ns)
        GLOBAL_SUM_WEIGHTS_NS = max(float(np.sum(GLOBAL_WEIGHTS_NS)), 1e-12)
    else:
        GLOBAL_DIFF_NS = None
        GLOBAL_LOG_NS = None
        GLOBAL_VAR_NS = 1.0
        GLOBAL_VAR_DIFF_NS = 1.0
        GLOBAL_VAR_LOG_NS = 1.0
        GLOBAL_NORM_NONLIN_NS = None
        GLOBAL_WEIGHTS_NS = None
        GLOBAL_SUM_WEIGHTS_NS = 1.0

    if fit_sat and cond_sat is not None:
        GLOBAL_DIFF_SAT = np.diff(cond_sat)
        GLOBAL_LOG_SAT = np.log10(np.clip(cond_sat, 5e-5, None))
        GLOBAL_VAR_SAT = max(np.var(cond_sat), 1e-6)
        GLOBAL_VAR_DIFF_SAT = max(np.var(GLOBAL_DIFF_SAT), 1e-9)
        GLOBAL_VAR_LOG_SAT = max(np.var(GLOBAL_LOG_SAT), 1e-6)
        GLOBAL_NORM_NONLIN_SAT = compute_normalized_nonlinearity(pulses_sat, cond_sat, raw_x=raw_p_sat, raw_y=raw_c_sat)
        GLOBAL_WEIGHTS_SAT = compute_nonlinearity_weights(pulses_sat, cond_sat, w_base=0.50, raw_x=raw_p_sat, raw_y=raw_c_sat)
        GLOBAL_SUM_WEIGHTS_SAT = max(float(np.sum(GLOBAL_WEIGHTS_SAT)), 1e-12)
    else:
        GLOBAL_DIFF_SAT = None
        GLOBAL_LOG_SAT = None
        GLOBAL_VAR_SAT = 1.0
        GLOBAL_VAR_DIFF_SAT = 1.0
        GLOBAL_VAR_LOG_SAT = 1.0
        GLOBAL_NORM_NONLIN_SAT = None
        GLOBAL_WEIGHTS_SAT = None
        GLOBAL_SUM_WEIGHTS_SAT = 1.0

    if fit_122 and cond_122 is not None:
        GLOBAL_DIFF_122 = np.diff(cond_122)
        GLOBAL_LOG_122 = np.log10(np.clip(cond_122, 5e-5, None))
        GLOBAL_VAR_122 = max(np.var(cond_122), 1e-6)
        GLOBAL_VAR_DIFF_122 = max(np.var(GLOBAL_DIFF_122), 1e-9)
        GLOBAL_VAR_LOG_122 = max(np.var(GLOBAL_LOG_122), 1e-6)
        GLOBAL_NORM_NONLIN_122 = compute_normalized_nonlinearity(pulses_122, cond_122, raw_x=raw_p_122, raw_y=raw_c_122)
        GLOBAL_WEIGHTS_122 = compute_nonlinearity_weights(pulses_122, cond_122, w_base=0.50, raw_x=raw_p_122, raw_y=raw_c_122)
        GLOBAL_SUM_WEIGHTS_122 = max(float(np.sum(GLOBAL_WEIGHTS_122)), 1e-12)
    else:
        GLOBAL_DIFF_122 = None
        GLOBAL_LOG_122 = None
        GLOBAL_VAR_122 = 1.0
        GLOBAL_VAR_DIFF_122 = 1.0
        GLOBAL_VAR_LOG_122 = 1.0
        GLOBAL_NORM_NONLIN_122 = None
        GLOBAL_WEIGHTS_122 = None
        GLOBAL_SUM_WEIGHTS_122 = 1.0

    # 1. Non-saturating Sweep Simulator
    GLOBAL_SIM_NS = MolmemSimulator(device='cpu', backend='numba')
    GLOBAL_SIM_NS.detailedPrint = False
    GLOBAL_SIM_NS.addCrossbarMatrix(device_type=base_params, multiThread=False, rows=1, cols=1)
    
    t_switch_ns = (np.max(pulses_ns) / 2.0) * t_period
    def v_pulse_train_ns(t, sources):
        if isinstance(t, np.ndarray):
            out = np.zeros_like(t)
            mask = t < t_switch_ns
            if np.any(mask):
                out[mask] = sources[0].getVoltage(t[mask])
            if np.any(~mask):
                out[~mask] = sources[1].getVoltage(t[~mask])
            return out
        return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
    edge_points_ns = [t_switch_ns]
    GLOBAL_SIM_NS.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=edge_points_ns)
    GLOBAL_SIM_NS.addDcSource(col=1, voltage=0.0)

    # 2. Saturating Sweep Simulator
    GLOBAL_SIM_SAT = MolmemSimulator(device='cpu', backend='numba')
    GLOBAL_SIM_SAT.detailedPrint = False
    GLOBAL_SIM_SAT.addCrossbarMatrix(device_type=base_params, multiThread=False, rows=1, cols=1)
    
    t_switch_sat = (np.max(pulses_sat) / 2.0) * t_period
    def v_pulse_train_sat(t, sources):
        if isinstance(t, np.ndarray):
            out = np.zeros_like(t)
            mask = t < t_switch_sat
            if np.any(mask):
                out[mask] = sources[0].getVoltage(t[mask])
            if np.any(~mask):
                out[~mask] = sources[1].getVoltage(t[~mask])
            return out
        return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
    edge_points_sat = [t_switch_sat]
    GLOBAL_SIM_SAT.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=edge_points_sat)
    GLOBAL_SIM_SAT.addDcSource(col=1, voltage=0.0)

    # 3. 1.22V Saturating Sweep Simulator
    GLOBAL_SIM_122 = MolmemSimulator(device='cpu', backend='numba')
    GLOBAL_SIM_122.detailedPrint = False
    GLOBAL_SIM_122.addCrossbarMatrix(device_type=base_params, multiThread=False, rows=1, cols=1)
    
    t_switch_122 = (np.max(pulses_122) / 2.0) * t_period
    GLOBAL_VPOT_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
    GLOBAL_VDEP_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    def v_pulse_train_122(t, sources):
        if isinstance(t, np.ndarray):
            out = np.zeros_like(t)
            mask = t < t_switch_122
            if np.any(mask):
                out[mask] = sources[0].getVoltage(t[mask])
            if np.any(~mask):
                out[~mask] = sources[1].getVoltage(t[~mask])
            return out
        return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
    edge_points_122 = [t_switch_122]
    GLOBAL_SIM_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[GLOBAL_VPOT_122, GLOBAL_VDEP_122], edge_points=edge_points_122)
    GLOBAL_SIM_122.addDcSource(col=1, voltage=0.0)

def update_simulator_candidate(sim, candidate):
    # Update device attributes in-place
    dev = sim.devices[0]['dev']
    dev.vSmooth = candidate["vSmooth"]
    dev.nSmooth = candidate["nSmooth"]
    dev.kappa = candidate["kappa"]
    dev.alpha = candidate["alpha"]
    dev.gScale = candidate["gScale"]
    dev.vTh = candidate["vTh"]
    dev.vRefPos = candidate["vRefPos"]
    dev.vRefNeg = candidate["vRefNeg"]
    dev.kDischarge = candidate["kDischarge"]
    dev.rC = candidate["rC"]
    dev.E_a = candidate.get("E_a", 0.0)
    dev.R_th = candidate.get("R_th", 0.0)
    dev.tau_th = candidate.get("tau_th", 1e-9)
    dev.gamma = candidate.get("gamma", 0.0)
    dev.beta_alpha = candidate.get("beta_alpha", 0.0)
    dev.beta_s = candidate.get("beta_s", 0.0)
    dev.rateAsymmetry = candidate.get("rate_asymmetry", 1.0)
    dev.rTopBlend = candidate.get("rTopBlend", 0.05)
    dev.rLatency = candidate.get("rLatency", 0.5)
    dev.fDischarge = candidate.get("fDischarge", 0.25)
    dev.cDischarge = candidate.get("cDischarge", 20.0)
    dev.rBottomBlend = candidate.get("rBottomBlend", 0.05)
    dev.f22StartVal = candidate["f22StartVal"]
    dev.f22PeakVal = candidate["f22PeakVal"]
    dev.f22EndVal = candidate["f22EndVal"]
    dev.f22PotScaleInv = 1.0 / (candidate["f22PeakVal"] - candidate["f22StartVal"])
    dev.f22DepScaleInv = 1.0 / (candidate["f22PeakVal"] - candidate["f22EndVal"])
    
    # Pre-calculate vBypass
    vref_max = max(abs(dev.vRefPos), abs(dev.vRefNeg))
    denom = max(1e-6, vref_max - dev.vTh + 1e-6)
    ratio = 1e-11 / (1e-8 * max(1e-6, dev.kappa))
    term1 = ratio ** (1.0 / max(0.1, dev.alpha))
    arg = (denom / max(1e-6, dev.vSmooth)) * term1
    if arg > 1e-20:
        vBypass_val = dev.vTh + dev.vSmooth * np.log(arg)
        dev.vBypass = max(0.0, min(dev.vTh - 3.0 * dev.vSmooth, vBypass_val))
    else:
        dev.vBypass = 0.0
        
    # Overwrite the JIT Structure-of-Arrays parameter cache directly if already initialized
    if hasattr(sim, '_soa_vSmooth_arr'):
        sim._soa_vSmooth_arr[0] = dev.vSmooth
        sim._soa_nSmooth_arr[0] = dev.nSmooth
        sim._soa_kappa_arr[0] = dev.kappa
        sim._soa_alpha_arr[0] = dev.alpha
        sim._soa_vTh_arr[0] = dev.vTh
        sim._soa_vBypass_arr[0] = dev.vBypass
        sim._soa_vRefPos_arr[0] = dev.vRefPos
        sim._soa_vRefNeg_arr[0] = dev.vRefNeg
        sim._soa_nMax_arr[0] = dev.nMax
        sim._soa_kDischarge_arr[0] = dev.kDischarge
        sim._soa_gScale_arr[0] = dev.gScale
        sim._soa_rC_arr[0] = dev.rC
        sim._soa_f22StartVal_arr[0] = dev.f22StartVal
        sim._soa_f22PeakVal_arr[0] = dev.f22PeakVal
        sim._soa_f22EndVal_arr[0] = dev.f22EndVal
        sim._soa_f22PotScaleInv_arr[0] = dev.f22PotScaleInv
        sim._soa_f22DepScaleInv_arr[0] = dev.f22DepScaleInv
        sim._soa_E_a_arr[0] = dev.E_a
        sim._soa_R_th_arr[0] = dev.R_th
        sim._soa_tau_th_arr[0] = dev.tau_th
        sim._soa_gamma_arr[0] = dev.gamma
        sim._soa_beta_alpha_arr[0] = dev.beta_alpha
        sim._soa_beta_s_arr[0] = dev.beta_s
        sim._soa_rateAsymmetry_arr[0] = dev.rateAsymmetry
        if hasattr(sim, '_soa_rTopBlend_arr'):
            sim._soa_rTopBlend_arr[0] = dev.rTopBlend
        if hasattr(sim, '_soa_fDischarge_arr'):
            sim._soa_fDischarge_arr[0] = dev.fDischarge
        if hasattr(sim, '_soa_cDischarge_arr'):
            sim._soa_cDischarge_arr[0] = dev.cDischarge
        if hasattr(sim, '_soa_rBottomBlend_arr'):
            sim._soa_rBottomBlend_arr[0] = dev.rBottomBlend
        if hasattr(sim, '_soa_rLatency_arr'):
            sim._soa_rLatency_arr[0] = dev.rLatency
    
    # Invalidate GPU physical parameter tensor cache
    if hasattr(sim, '_cached_phys_tensor'):
        delattr(sim, '_cached_phys_tensor')

# =============================================================================
# OPTIMIZATION HYPERPARAMETERS
# =============================================================================
POP_SIZE = 4000       # Population size multiplier for Differential Evolution
MAX_ITER = 50      # Maximum number of generations for Differential Evolution
TOL = 0.0          # Convergence tolerance for Differential Evolution (0.0 disables population-based early termination)
WORKERS = min(os.cpu_count() or 1, 61)  # Capped at 61 due to Windows WaitForMultipleObjects handle limit (max 64 minus 3 internal)
LBFGS_MAX_ITER = 20 # Maximum iterations for local L-BFGS-B fine-tuning

# Toggles for search constraints and empirical scaling
ENABLE_PREWARMING_SWEEP = False  # Set to True to enable initial LHS + Profile Likelihood boundary contraction
USE_RAW_EMPIRICAL_RANGE = True   # Set to True to use raw physical conductance (df["Y_value"]/0.5) without default model scaling

# Pre-optimization Conditioning Run Hyperparameters
CONDITIONING_SWEEP_SIZE = POP_SIZE*20  # Size of the initial random sweep population for ROI detection
PROFILE_SWEEP_STEPS = 200       # Grid step resolution per parameter for 1D profile likelihood sweeps
PROFILE_COST_THRESHOLD = 20.0   # Floor of the cost threshold for boundary contraction
PROFILE_SAFETY_MARGIN = 0.20    # Safety margin multiplier on each side of contracted parameter bounds
PROFILE_LHS_BLEND = 0.1         # LHS blending ratio (30% LHS / 70% random) for initial conditioning sweep
DE_LHS_BLEND = 0.1              # LHS blending ratio (30% LHS / 70% random) for DE population seeding
REFINEMENT_SWEEP_STEPS = 1500     # Grid step resolution for in-loop coordinate refinement sweep (0 disables it)

# Cost function formulation: Pure point-by-point distance MSE
# 0.0 cost represents 100% exact overlap across all time/pulse points
USE_NORMALIZED_MSE = True  # True: normalized by target variance (NMSE) for dimensionless balance across sweeps
GSCALE_NOMINAL = 6.30  # Empirical baseline
GSCALE_PRIOR_WEIGHT = 0.02 # Soft Bayesian/Tikhonov prior weight penalizing gScale deviation from nominal baseline


# =============================================================================
# Parameter Space Scaling Maps (Log10 vs. Linear)
# =============================================================================
LOG_SCALE_PARAMS = [
    True,   # 0: kappa (1e5 to 1e8)
    False,  # 1: alpha
    False,  # 2: vTh
    False,  # 3: gScale (originally index 5)
    False,  # 4: rC (originally index 6)
    False,  # 5: vSmooth (originally index 7)
    True,   # 6: nSmooth (originally index 8, 150.0 to 1500.0)
    True,   # 7: kDischarge (0.1 to 50.0: apex rapid drop boost amplitude)
    False,  # 8: E_a (originally index 13)
    True,   # 9: R_th (originally index 14, 100.0 to 1e5)
    True,   # 10: tau_th (originally index 15, 1e-9 to 3e-7)
    False,  # 11: gamma (originally index 16)
    False,  # 12: beta_alpha (originally index 17)
    False,  # 13: beta_s (originally index 19)
    False,  # 14: rate_asymmetry (0.05 to 3.0)
    False,  # 15: fDischarge (dynamic knee transition: 0.10 to 0.70)
    False,  # 16: rLatency (0.0 to 0.95: shelf latency throttle factor)
    False   # 17: rBottomBlend (0.10 to 1.90: saturation recovery base landing boundary scalar)
]

PARAM_NAMES = [
    "kappa", "alpha", "vTh", "gScale", "rC", 
    "vSmooth", "nSmooth", "kDischarge",
    "E_a", "R_th", "tau_th", "gamma", "beta_alpha", "beta_s",
    "rate_asymmetry", "fDischarge", "rLatency", "rBottomBlend"
]

def to_physical_params(x):
    x_phys = np.copy(x)
    for idx, is_log in enumerate(LOG_SCALE_PARAMS):
        if is_log:
            x_phys[idx] = 10**x[idx]
    return x_phys


# =============================================================================
# Dashboard Console Display Utilities (Always-at-the-Bottom Status Bar)
# =============================================================================
GLOBAL_DASHBOARD_STATUS = ""

def print_log(message):
    import shutil
    global GLOBAL_DASHBOARD_STATUS
    sys.stdout.write("\r\033[K" + message + "\n")
    if GLOBAL_DASHBOARD_STATUS:
        cols = shutil.get_terminal_size().columns
        sys.stdout.write("\r\033[K" + GLOBAL_DASHBOARD_STATUS[:cols - 1])
    sys.stdout.flush()

def update_dashboard(phase, current, total, cost, best_xk, dither=None, w_base=None, avg_k=None):
    import shutil
    global GLOBAL_DASHBOARD_STATUS
    bar_width = 20
    percent = int(current / total * 100) if total > 0 else 0
    filled = int(current / total * bar_width) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    
    x_phys = to_physical_params(best_xk)
    kp = x_phys[0]
    al = x_phys[1]
    vTh = x_phys[2]
    kDisch = x_phys[7]
    E_a = x_phys[8]
    R_th = x_phys[9]
    gamma = x_phys[11]
    if len(x_phys) >= 19:
        uKnee = x_phys[16]
        rLat = x_phys[17]
        rBot = x_phys[18]
    elif len(x_phys) >= 18:
        uKnee = x_phys[16]
        rLat = x_phys[17]
        rBot = 1.0
    elif len(x_phys) >= 17:
        uKnee = 0.35
        rLat = x_phys[16]
        rBot = 1.0
    else:
        uKnee = 0.35
        rLat = 0.5
        rBot = 1.0
    k_str_pop = f" kDisch: avg={avg_k:.2f}|best={kDisch:.2f}" if avg_k is not None else f" kDisch={kDisch:.2f}"
    p_str = f"kp={kp:.2e} al={al:.2f}{k_str_pop} uKn={uKnee:.2f} rLat={rLat:.2f} rBot={rBot:.2f} Ea={E_a:.2f} g={gamma:.2f}"
    
    if phase == 1:
        dither_str = f" | dither: {dither[1]:.2f} (0.30)" if dither is not None else ""
        w_base_str = f" | w_base: {w_base:.2f}" if w_base is not None else ""
        GLOBAL_DASHBOARD_STATUS = f"Phase 1/2: DE |{bar}| {percent}% (Gen {current}/{total}) | Best Cost: {cost:.6f}{dither_str}{w_base_str} | {p_str}"
    else:
        GLOBAL_DASHBOARD_STATUS = f"Phase 2/2: L-BFGS-B |{bar}| {percent}% (Basin {current}/{total}) | Best Cost: {cost:.6f} | {p_str}"
        
    cols = shutil.get_terminal_size().columns
    sys.stdout.write("\r\033[K" + GLOBAL_DASHBOARD_STATUS[:cols - 1])
    sys.stdout.flush()

def update_sweep_dashboard(phase_name, current, total, best_cost):
    import shutil
    global GLOBAL_DASHBOARD_STATUS
    bar_width = 20
    percent = int(current / total * 100) if total > 0 else 0
    filled = int(current / total * bar_width) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    
    GLOBAL_DASHBOARD_STATUS = f"Conditioning: {phase_name} |{bar}| {percent}% ({current}/{total}) | Best Cost: {best_cost:.6f}"
    cols = shutil.get_terminal_size().columns
    sys.stdout.write("\r\033[K" + GLOBAL_DASHBOARD_STATUS[:cols - 1])
    sys.stdout.flush()

def clear_dashboard():
    global GLOBAL_DASHBOARD_STATUS
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    GLOBAL_DASHBOARD_STATUS = ""


def to_optimizer_params(x_phys):
    x_opt = np.copy(x_phys)
    for idx, is_log in enumerate(LOG_SCALE_PARAMS):
        if is_log:
            x_opt[idx] = np.log10(x_phys[idx])
    return x_opt

# =============================================================================
# Global Helper: Build dictionary of candidate parameters
# =============================================================================
def get_dict_params(x_opt, base_params):
    x_phys = to_physical_params(x_opt)
    d = base_params.copy()
    d["kappa"] = x_phys[0]
    d["alpha"] = x_phys[1]
    d["vTh"] = x_phys[2]
    d["vRefPos"] = 0.9
    d["vRefNeg"] = 0.75
    d["gScale"] = x_phys[3]
    d["rC"] = x_phys[4]
    d["vSmooth"] = x_phys[5]
    d["nSmooth"] = x_phys[6]
    d["kDischarge"] = x_phys[7]
    d["f22StartVal"] = base_params["f22StartVal"]
    d["f22PeakVal"] = base_params["f22PeakVal"]
    d["f22EndVal"] = base_params["f22EndVal"]
    d["E_a"] = x_phys[8]
    d["R_th"] = x_phys[9]
    d["tau_th"] = x_phys[10]
    d["gamma"] = x_phys[11]
    d["beta_alpha"] = x_phys[12]
    d["beta_s"] = x_phys[13]
    d["rate_asymmetry"] = x_phys[14]
    if len(x_phys) >= 18:
        d["fDischarge"] = x_phys[15]
        d["rLatency"] = x_phys[16]
        d["rBottomBlend"] = x_phys[17]
    elif len(x_phys) >= 17:
        d["fDischarge"] = x_phys[15]
        d["rLatency"] = x_phys[16]
        d["rBottomBlend"] = base_params.get("rBottomBlend", 1.0)
    elif len(x_phys) >= 16:
        d["rLatency"] = x_phys[15]
        d["fDischarge"] = base_params.get("fDischarge", 0.35)
        d["rBottomBlend"] = base_params.get("rBottomBlend", 1.0)
    else:
        d["rLatency"] = 0.5
        d["fDischarge"] = base_params.get("fDischarge", 0.35)
        d["rBottomBlend"] = base_params.get("rBottomBlend", 1.0)
    return d

def save_preset_txt(x, base_params, txt_path, func_name, device_name, data_dir):
    final_params = get_dict_params(x, base_params)
    n_init_best = 0.0
    preset_str = f"""def {func_name}(data_dir=None):
    \"\"\"
    Returns optimized parameters and PWL paths for the {device_name} device (Multi-Sweep Fit).
    \"\"\"
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        
    return {{
        "vSmooth": {final_params['vSmooth']:.6f},
        "nSmooth": {final_params['nSmooth']:.2f},
        "kappa": {final_params['kappa']:.6e},
        "alpha": {final_params['alpha']:.6f},
        "gScale": {final_params['gScale']:.6f},
        "vTh": {final_params['vTh']:.6f},
        "vRefPos": {final_params['vRefPos']:.6f},
        "vRefNeg": {final_params['vRefNeg']:.6f},
        "nMax": {final_params['nMax']},
        "nInit": {n_init_best:.6f},
        "kDischarge": {final_params['kDischarge']:.2f},
        "rC": {final_params['rC']:.6f},
        "E_a": {final_params['E_a']:.6f},
        "R_th": {final_params['R_th']:.2f},
        "tau_th": {final_params['tau_th']:.3e},
        "gamma": {final_params['gamma']:.6f},
        "beta_alpha": {final_params['beta_alpha']:.6f},
        "beta_s": {final_params['beta_s']:.6f},
        "rate_asymmetry": {final_params['rate_asymmetry']:.6f},
        "rTopBlend": {final_params.get('rTopBlend', 0.05):.6f},
        "fDischarge": {final_params.get('fDischarge', 0.25):.6f},
        "rBottomBlend": {final_params.get('rBottomBlend', 0.05):.6f},
        "rLatency": {final_params.get('rLatency', 0.5):.6f},
        "k_overdrive": {final_params.get('k_overdrive', 2.85):.6f},
        "safe_max_vwrite": 0.00,
        "f22StartVal": {final_params['f22StartVal']:.8f},
        "f22PeakVal": {final_params['f22PeakVal']:.8f},
        "f22EndVal": {final_params['f22EndVal']:.8f},
        # Lookup table files
        "iScale_file": os.path.join(data_dir, "I_scale.pwl"),
        "vScale_file": os.path.join(data_dir, "V_scale.pwl"),
        "f22PotMap_file": os.path.join(data_dir, "f22_pot_map.pwl"),
        "f22DepMap_file": os.path.join(data_dir, "f22_dep_map.pwl")
    }}
"""
    with open(txt_path, "w") as f:
        f.write(preset_str)

# =============================================================================
# Cost Function using the True MolmemSimulator Solver
# =============================================================================
def cost_function(cand_input):
    cohort_w_base = None
    if isinstance(cand_input, tuple) and len(cand_input) == 2 and isinstance(cand_input[1], (int, float)):
        x, cohort_w_base = cand_input
    else:
        x = cand_input

    global GLOBAL_COND_NS, GLOBAL_PULSES_NS, GLOBAL_COND_SAT, GLOBAL_PULSES_SAT
    global GLOBAL_COND_122, GLOBAL_PULSES_122, GLOBAL_BASE_PARAMS
    global GLOBAL_T_PERIOD, GLOBAL_T_WIDTH, GLOBAL_VPOT, GLOBAL_VDEP
    global GLOBAL_SIM_NS, GLOBAL_SIM_SAT, GLOBAL_SIM_122
    global GLOBAL_DIFF_NS, GLOBAL_DIFF_SAT, GLOBAL_DIFF_122
    global GLOBAL_VAR_DIFF_NS, GLOBAL_VAR_DIFF_SAT, GLOBAL_VAR_DIFF_122
    global GLOBAL_VAR_NS, GLOBAL_VAR_SAT, GLOBAL_VAR_122
    global GLOBAL_LOG_NS, GLOBAL_LOG_SAT, GLOBAL_LOG_122
    global GLOBAL_SIM_TOL_VAL, GLOBAL_BEST_COST_VAL
    global GLOBAL_FIT_NS, GLOBAL_FIT_SAT, GLOBAL_FIT_122
    global GLOBAL_WEIGHTS_NS, GLOBAL_WEIGHTS_SAT, GLOBAL_WEIGHTS_122
    global GLOBAL_SUM_WEIGHTS_NS, GLOBAL_SUM_WEIGHTS_SAT, GLOBAL_SUM_WEIGHTS_122
    global GLOBAL_NORM_NONLIN_NS, GLOBAL_NORM_NONLIN_SAT, GLOBAL_NORM_NONLIN_122
    global GLOBAL_LAST_G_NS, GLOBAL_LAST_G_SAT, GLOBAL_LAST_G_122
    
    candidate = get_dict_params(x, GLOBAL_BASE_PARAMS)
    
    # Retrieve tolerance from shared value if available
    sim_tol = GLOBAL_SIM_TOL_VAL.value if GLOBAL_SIM_TOL_VAL is not None else 1e-3
    
    # 1. Non-saturating sweep (33k pulses: 16.5k pot, 16.5k dep)
    if GLOBAL_FIT_NS:
        update_simulator_candidate(GLOBAL_SIM_NS, candidate)
        GLOBAL_SIM_NS.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
        GLOBAL_SIM_NS.devices[0]['dev']._forceUpdateConductanceParams()
        
        t_stop_ns = np.max(GLOBAL_PULSES_NS) * GLOBAL_T_PERIOD
        
        try:
            GLOBAL_SIM_NS.run(tEnd=t_stop_ns, min_dt=1e-11, tol=sim_tol, title=None, recordHistory=['g', 'state', 'f22'])
            t_arr_ns = GLOBAL_SIM_NS.getTime()
            g_sim_ns = GLOBAL_SIM_NS.getConductance(row=1, col=1)
            g_sim_ns_ms = g_sim_ns * 1e3
            sim_pulses_ns = t_arr_ns / GLOBAL_T_PERIOD
            g_sim_ns_interp = np.interp(GLOBAL_PULSES_NS - 0.05, sim_pulses_ns, g_sim_ns_ms)
            GLOBAL_LAST_G_NS = np.copy(g_sim_ns_interp)
            
            # Curvature-weighted point-to-point distance MSE
            err_sq_ns = (g_sim_ns_interp - GLOBAL_COND_NS) ** 2
            if cohort_w_base is not None and GLOBAL_NORM_NONLIN_NS is not None:
                w_ns = cohort_w_base + (1.0 - cohort_w_base) * GLOBAL_NORM_NONLIN_NS
                sum_w_ns = max(float(np.sum(w_ns)), 1e-12)
            else:
                w_ns = GLOBAL_WEIGHTS_NS
                sum_w_ns = GLOBAL_SUM_WEIGHTS_NS

            if USE_NORMALIZED_MSE:
                cost_ns = np.sum(w_ns * err_sq_ns) / (sum_w_ns * GLOBAL_VAR_NS)
            else:
                cost_ns = np.sum(w_ns * err_sq_ns) / sum_w_ns

            # Derivative / slope match + endpoint boundary pinning + apex peak match
            if GLOBAL_DIFF_NS is not None:
                cost_deriv_ns = np.mean((np.diff(g_sim_ns_interp) - GLOBAL_DIFF_NS) ** 2) / GLOBAL_VAR_DIFF_NS
                cost_endpoints_ns = ((g_sim_ns_interp[0] - GLOBAL_COND_NS[0]) ** 2 + 3.0 * (g_sim_ns_interp[-1] - GLOBAL_COND_NS[-1]) ** 2) / GLOBAL_VAR_NS
                cost_peak_ns = ((np.max(g_sim_ns_interp) - np.max(GLOBAL_COND_NS)) ** 2) / GLOBAL_VAR_NS
                cost_ns += 0.25 * cost_deriv_ns + 1.50 * cost_endpoints_ns + 0.50 * cost_peak_ns
        except Exception:
            cost_ns = 1e6
    else:
        cost_ns = 0.0

    # Math-Barrier Pruning check after sweep 1
    if GLOBAL_BEST_COST_VAL is not None and cost_ns > GLOBAL_BEST_COST_VAL.value:
        return cost_ns

    # 2. Saturating sweep (52k pulses: 26k pot, 26k dep)
    if GLOBAL_FIT_SAT:
        update_simulator_candidate(GLOBAL_SIM_SAT, candidate)
        GLOBAL_SIM_SAT.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
        GLOBAL_SIM_SAT.devices[0]['dev']._forceUpdateConductanceParams()
        
        t_stop_sat = np.max(GLOBAL_PULSES_SAT) * GLOBAL_T_PERIOD
        
        try:
            GLOBAL_SIM_SAT.run(tEnd=t_stop_sat, min_dt=1e-11, tol=sim_tol, title=None, recordHistory=['g', 'state', 'f22'])
            t_arr_sat = GLOBAL_SIM_SAT.getTime()
            g_sim_sat = GLOBAL_SIM_SAT.getConductance(row=1, col=1)
            g_sim_sat_ms = g_sim_sat * 1e3
            sim_pulses_sat = t_arr_sat / GLOBAL_T_PERIOD
            g_sim_sat_interp = np.interp(GLOBAL_PULSES_SAT - 0.05, sim_pulses_sat, g_sim_sat_ms)
            GLOBAL_LAST_G_SAT = np.copy(g_sim_sat_interp)
            
            # Curvature-weighted point-to-point distance MSE
            err_sq_sat = (g_sim_sat_interp - GLOBAL_COND_SAT) ** 2
            if cohort_w_base is not None and GLOBAL_NORM_NONLIN_SAT is not None:
                w_sat = cohort_w_base + (1.0 - cohort_w_base) * GLOBAL_NORM_NONLIN_SAT
                sum_w_sat = max(float(np.sum(w_sat)), 1e-12)
            else:
                w_sat = GLOBAL_WEIGHTS_SAT
                sum_w_sat = GLOBAL_SUM_WEIGHTS_SAT

            if USE_NORMALIZED_MSE:
                cost_sat = np.sum(w_sat * err_sq_sat) / (sum_w_sat * GLOBAL_VAR_SAT)
            else:
                cost_sat = np.sum(w_sat * err_sq_sat) / sum_w_sat

            # Derivative / slope match + endpoint boundary pinning + apex peak match
            if GLOBAL_DIFF_SAT is not None:
                cost_deriv_sat = np.mean((np.diff(g_sim_sat_interp) - GLOBAL_DIFF_SAT) ** 2) / GLOBAL_VAR_DIFF_SAT
                cost_endpoints_sat = ((g_sim_sat_interp[0] - GLOBAL_COND_SAT[0]) ** 2 + 3.0 * (g_sim_sat_interp[-1] - GLOBAL_COND_SAT[-1]) ** 2) / GLOBAL_VAR_SAT
                cost_peak_sat = ((np.max(g_sim_sat_interp) - np.max(GLOBAL_COND_SAT)) ** 2) / GLOBAL_VAR_SAT
                cost_sat += 0.25 * cost_deriv_sat + 1.50 * cost_endpoints_sat + 0.50 * cost_peak_sat

            # Saturation barrier penalty: disincentivize optimizer from staying below n_base (16520.0)
            n_arr_sat = GLOBAL_SIM_SAT.getState(row=1, col=1)
            if n_arr_sat is not None and len(n_arr_sat) > 0:
                n_peak_sat = float(np.max(n_arr_sat))
                if n_peak_sat < 16520.0:
                    shortfall = (16520.0 - n_peak_sat) / 16520.0
                    cost_sat += 5.0 * (shortfall ** 2)
        except Exception:
            cost_sat = 1e6
    else:
        cost_sat = 0.0

    cost_ns_sat = cost_ns + cost_sat

    # Math-Barrier Pruning check after sweep 2
    if GLOBAL_BEST_COST_VAL is not None and cost_ns_sat > GLOBAL_BEST_COST_VAL.value:
        return cost_ns_sat

    # 3. 1.22V Saturating sweep (33,040 pulses: 16,520 pot at 1.22V, 16,520 dep at -0.75V)
    if GLOBAL_FIT_122:
        update_simulator_candidate(GLOBAL_SIM_122, candidate)
        GLOBAL_SIM_122.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
        GLOBAL_SIM_122.devices[0]['dev']._forceUpdateConductanceParams()
        
        t_stop_122 = np.max(GLOBAL_PULSES_122) * GLOBAL_T_PERIOD
        
        try:
            GLOBAL_SIM_122.run(tEnd=t_stop_122, min_dt=1e-11, tol=sim_tol, title=None, recordHistory=['g', 'state', 'f22'])
            t_arr_122 = GLOBAL_SIM_122.getTime()
            g_sim_122 = GLOBAL_SIM_122.getConductance(row=1, col=1)
            g_sim_122_ms = g_sim_122 * 1e3
            sim_pulses_122 = t_arr_122 / GLOBAL_T_PERIOD
            g_sim_122_interp = np.interp(GLOBAL_PULSES_122 - 0.05, sim_pulses_122, g_sim_122_ms)
            GLOBAL_LAST_G_122 = np.copy(g_sim_122_interp)
            
            # Curvature-weighted point-to-point distance MSE
            err_sq_122 = (g_sim_122_interp - GLOBAL_COND_122) ** 2
            if cohort_w_base is not None and GLOBAL_NORM_NONLIN_122 is not None:
                w_122 = cohort_w_base + (1.0 - cohort_w_base) * GLOBAL_NORM_NONLIN_122
                sum_w_122 = max(float(np.sum(w_122)), 1e-12)
            else:
                w_122 = GLOBAL_WEIGHTS_122
                sum_w_122 = GLOBAL_SUM_WEIGHTS_122

            if USE_NORMALIZED_MSE:
                cost_122 = np.sum(w_122 * err_sq_122) / (sum_w_122 * GLOBAL_VAR_122)
            else:
                cost_122 = np.sum(w_122 * err_sq_122) / sum_w_122

            # Derivative / slope match + endpoint boundary pinning + apex peak match
            if GLOBAL_DIFF_122 is not None:
                cost_deriv_122 = np.mean((np.diff(g_sim_122_interp) - GLOBAL_DIFF_122) ** 2) / GLOBAL_VAR_DIFF_122
                cost_endpoints_122 = ((g_sim_122_interp[0] - GLOBAL_COND_122[0]) ** 2 + 3.0 * (g_sim_122_interp[-1] - GLOBAL_COND_122[-1]) ** 2) / GLOBAL_VAR_122
                cost_peak_122 = ((np.max(g_sim_122_interp) - np.max(GLOBAL_COND_122)) ** 2) / GLOBAL_VAR_122
                cost_122 += 0.25 * cost_deriv_122 + 1.50 * cost_endpoints_122 + 0.50 * cost_peak_122
        except Exception:
            cost_122 = 1e6
    else:
        cost_122 = 0.0
        
    combined_cost = cost_ns_sat + cost_122
    
    # Soft Bayesian prior on gScale to break collinearity with kappa and avoid high-gScale drift
    if GSCALE_PRIOR_WEIGHT > 0.0 and GSCALE_NOMINAL > 0.0:
        g_scale_cand = float(candidate.get("gScale", GSCALE_NOMINAL))
        g_scale_dev = (g_scale_cand - GSCALE_NOMINAL) / GSCALE_NOMINAL
        cost_prior_gscale = GSCALE_PRIOR_WEIGHT * (g_scale_dev ** 2)
        combined_cost += cost_prior_gscale

    # Update global best cost atomically
    if GLOBAL_BEST_COST_VAL is not None:
        with GLOBAL_BEST_COST_VAL.get_lock():
            if combined_cost < GLOBAL_BEST_COST_VAL.value:
                GLOBAL_BEST_COST_VAL.value = combined_cost
                
    global GLOBAL_LAST_X, GLOBAL_LAST_COST
    GLOBAL_LAST_X = np.copy(x)
    GLOBAL_LAST_COST = combined_cost
    if GLOBAL_FIT_NS and 'g_sim_ns_interp' in locals():
        GLOBAL_LAST_G_NS = np.copy(g_sim_ns_interp)
    if GLOBAL_FIT_SAT and 'g_sim_sat_interp' in locals():
        GLOBAL_LAST_G_SAT = np.copy(g_sim_sat_interp)
    if GLOBAL_FIT_122 and 'g_sim_122_interp' in locals():
        GLOBAL_LAST_G_122 = np.copy(g_sim_122_interp)
    
    return combined_cost


# =============================================================================
# Helper function to check and update/register device in devices.py
# =============================================================================
def register_or_update_device(device_name, final_params, n_init_best, devices_py_path, exists):
    func_name = device_name.replace(" ", "_").replace("-", "_")
    clean_func_name = func_name.lower()
    
    if not os.path.exists(devices_py_path):
        print(f"[Warning] Could not find devices.py at: {devices_py_path}")
        return
        
    with open(devices_py_path, "r") as f:
        content = f.read()
        
    new_func_str = f"""def {func_name}(data_dir=None):
    \"\"\"
    Returns optimized parameters and PWL paths for the {device_name} device (Multi-Sweep Fit).
    \"\"\"
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        
    return {{
        "vSmooth": {final_params['vSmooth']:.6f},
        "nSmooth": {final_params['nSmooth']:.2f},
        "kappa": {final_params['kappa']:.6e},
        "alpha": {final_params['alpha']:.6f},
        "gScale": {final_params['gScale']:.6f},
        "vTh": {final_params['vTh']:.6f},
        "vRefPos": {final_params['vRefPos']:.6f},
        "vRefNeg": {final_params['vRefNeg']:.6f},
        "nMax": {final_params['nMax']},
        "nInit": {n_init_best:.6f},
        "kDischarge": {final_params['kDischarge']:.2f},
        "rC": {final_params['rC']:.6f},
        "E_a": {final_params['E_a']:.6f},
        "R_th": {final_params['R_th']:.2f},
        "tau_th": {final_params['tau_th']:.3e},
        "gamma": {final_params['gamma']:.6f},
        "beta_alpha": {final_params['beta_alpha']:.6f},
        "beta_s": {final_params['beta_s']:.6f},
        "rate_asymmetry": {final_params.get('rate_asymmetry', 1.0):.8f},
        "rLatency": {final_params.get('rLatency', 0.5):.6f},
        "f22StartVal": {final_params['f22StartVal']:.8f},
        "f22PeakVal": {final_params['f22PeakVal']:.8f},
        "f22EndVal": {final_params['f22EndVal']:.8f},
        # Lookup table files
        "iScale_file": os.path.join(data_dir, "I_scale.pwl"),
        "vScale_file": os.path.join(data_dir, "V_scale.pwl"),
        "f22PotMap_file": os.path.join(data_dir, "f22_pot_map.pwl"),
        "f22DepMap_file": os.path.join(data_dir, "f22_dep_map.pwl")
    }}
"""
    
    if exists:
        start_idx = content.find(f"def {func_name}(")
        end_idx = len(content)
        next_def = content.find("def ", start_idx + 1)
        next_registry = content.find("DEVICE_REGISTRY", start_idx + 1)
        
        candidates = []
        if next_def != -1:
            candidates.append(next_def)
        if next_registry != -1:
            candidates.append(next_registry)
        if candidates:
            end_idx = min(candidates)
            
        new_content = content[:start_idx] + new_func_str + "\n" + content[end_idx:]
        print(f"Successfully updated '{func_name}' in devices.py.")
    else:
        registry_idx = content.find("DEVICE_REGISTRY")
        if registry_idx == -1:
            new_content = content + "\n\n" + new_func_str
        else:
            new_content = content[:registry_idx] + new_func_str + "\n\n" + content[registry_idx:]
            
        reg_start = new_content.find("DEVICE_REGISTRY = {")
        if reg_start != -1:
            reg_end = new_content.find("}", reg_start)
            if reg_end != -1:
                entry_line = f'    "{clean_func_name}": {func_name},\n'
                new_content = new_content[:reg_end] + entry_line + new_content[reg_end:]
        print(f"Successfully registered '{func_name}' as '{clean_func_name}' in devices.py.")
        
    with open(devices_py_path, "w") as f:
        f.write(new_content)


def make_parallel_jacobian(pool, cost_func, bounds, epsilon=1e-4):
    """
    Returns a function that computes the numerical gradient (jacobian) of cost_func
    in parallel using the provided multiprocessing Pool, utilizing central differences.
    """
    def parallel_jac(x):
        n = len(x)
        perturbed_xs = []
        h_vals = []
        
        for i in range(n):
            lo, hi = bounds[i]
            # Parameter-specific perturbation based on optimized bounds width
            h = epsilon * (hi - lo)
            h_vals.append(h)
            
            x_plus = x.copy()
            x_plus[i] += h
            perturbed_xs.append(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= h
            perturbed_xs.append(x_minus)
            
        # Distribute 2n evaluations in parallel across JIT-warmed pool workers
        results = pool.map(cost_func, perturbed_xs)
        
        grad = np.zeros(n)
        for i in range(n):
            y_plus = results[2 * i]
            y_minus = results[2 * i + 1]
            grad[i] = (y_plus - y_minus) / (2.0 * h_vals[i])
        return grad
    return parallel_jac


# =============================================================================
# Run Optimization Main
# =============================================================================
def main():
    os.environ["MOLMEM_ACTIVE_WORKER_TASK"] = "1"
    print("\n" + "="*64)
    print("  STAGE: Device Parameter Optimization (True Simulator Fit)")
    print("="*64 + "\n")
    
    try:
        if not ensure_molmem_core_sealed():
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[Attestation Warning] Verification error: {e}", flush=True)

    # Warm up JIT compilers in the main thread first to avoid concurrent compilation deadlocks
    print("Warming up JIT compilers...", flush=True)
    MolmemSimulator.warmup_compilers()
    print("Warmup complete. Initializing datasets...\n", flush=True)
    
    fitting_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(fitting_dir)
    data_dir = os.path.join(project_root, "PythonSimulator", "molmem_lib", "data")
    
    detected_devices = []
    for entry in os.scandir(fitting_dir):
        if entry.is_dir() and entry.name not in ["Figures", "__pycache__", "PlotData", "Screenshots", "TargetVsRawPlots"]:
            has_ns = False
            has_sat = False
            has_122 = False
            ns_file = ""
            sat_file = ""
            sat_122_file = ""
            for file in os.scandir(entry.path):
                if file.is_file() and file.name.lower().endswith(".csv"):
                    if "nonsaturation" in file.name.lower():
                        has_ns = True
                        ns_file = file.path
                    elif "1.22vsaturation" in file.name.lower():
                        has_122 = True
                        sat_122_file = file.path
                    elif "saturation" in file.name.lower():
                        has_sat = True
                        sat_file = file.path
            if has_ns and has_sat and has_122:
                detected_devices.append({
                    "name": entry.name,
                    "ns_file": ns_file,
                    "sat_file": sat_file,
                    "sat_122_file": sat_122_file
                })
                
    detected_devices.sort(key=lambda d: d["name"])
    
    if not detected_devices:
        print("[Error] No valid device subdirectories found.")
        sys.exit(1)
        
    print("Detected device subdirectories:")
    for idx, dev in enumerate(detected_devices):
        print(f" [{idx + 1}] {dev['name']}")
        
    if len(detected_devices) == 1:
        print(f"\nAuto-selecting the only available device: {detected_devices[0]['name']}")
        sel_idx = 0
    else:
        try:
            selection = input(f"\nSelect a device to fit (1-{len(detected_devices)}): ").strip()
            sel_idx = int(selection) - 1
            if sel_idx < 0 or sel_idx >= len(detected_devices):
                raise ValueError
        except (ValueError, KeyboardInterrupt, EOFError):
            print("Invalid selection. Exiting.")
            sys.exit(1)
        
    selected_device = detected_devices[sel_idx]
    device_name = selected_device["name"]
    csv_path_ns = selected_device["ns_file"]
    csv_path_sat = selected_device["sat_file"]
    csv_path_122 = selected_device["sat_122_file"]
    
    print(f"\nSelected device: {device_name}")
    
    # Selection of sweeps to fit
    print("\nSelect which datasets to include in the optimization:")
    print(" [1] Non-Saturating Sweep only")
    print(" [2] Saturating Sweep only")
    print(" [3] 1.22V Saturating Sweep only")
    print(" [4] Custom combination (comma-separated, e.g. 1,3)")
    print(" [5] All Sweeps (Default)")
    
    fit_ns = True
    fit_sat = True
    fit_122 = True
    selection_str = "ALL"
    
    try:
        ans = input("Enter selection (1-5) [5]: ").strip()
        if ans == "1":
            fit_ns, fit_sat, fit_122 = True, False, False
            selection_str = "NS"
        elif ans == "2":
            fit_ns, fit_sat, fit_122 = False, True, False
            selection_str = "SAT"
        elif ans == "3":
            fit_ns, fit_sat, fit_122 = False, False, True
            selection_str = "122V"
        elif ans == "4":
            sub_ans = input("Enter indices (e.g. 1,3): ").strip().split(",")
            fit_ns = "1" in sub_ans
            fit_sat = "2" in sub_ans
            fit_122 = "3" in sub_ans
            parts = []
            if fit_ns: parts.append("NS")
            if fit_sat: parts.append("SAT")
            if fit_122: parts.append("122V")
            selection_str = "_".join(parts) if parts else "ALL"
        else:
            fit_ns, fit_sat, fit_122 = True, True, True
            selection_str = "ALL"
    except (KeyboardInterrupt, EOFError):
        print("\nDefaulting to All Sweeps.")
        fit_ns, fit_sat, fit_122 = True, True, True
        selection_str = "ALL"
        
    print(f"Fitting target behavior: {selection_str}")
    
    # We never update devices.py for subset-specific fits to avoid contaminating presets
    update_registry_after_fit = False
    exists_in_registry = False
    print("Registry updates to devices.py are DISABLED for subset fitting to preserve default presets.")
    func_name = device_name.replace(" ", "_").replace("-", "_")
    clean_func_name = func_name.lower()
        
    print(f"Loading empirical non-saturation dataset from {csv_path_ns}...")
    df_ns = pd.read_csv(csv_path_ns, comment='#')
    pulses_exp_ns_raw = df_ns["X_value"] * 330.4
    x_ns_min = pulses_exp_ns_raw.min()
    x_ns_max = pulses_exp_ns_raw.max()
    raw_pulses_exp_ns = (pulses_exp_ns_raw - x_ns_min) / (x_ns_max - x_ns_min) * 33040.0
    cond_exp_ns_ms_raw = df_ns["Y_value"].copy()
    
    print(f"Loading empirical saturation dataset from {csv_path_sat}...")
    df_sat = pd.read_csv(csv_path_sat, comment='#')
    pulses_exp_sat_raw = df_sat["X_value"] * 520.0
    x_sat_min = pulses_exp_sat_raw.min()
    x_sat_max = pulses_exp_sat_raw.max()
    raw_pulses_exp_sat = (pulses_exp_sat_raw - x_sat_min) / (x_sat_max - x_sat_min) * 52000.0
    cond_exp_sat_ms_raw = df_sat["Y_value"].copy()
    
    print(f"Loading empirical 1.22V saturation dataset from {csv_path_122}...")
    df_122 = pd.read_csv(csv_path_122, comment='#')
    pulses_exp_122_raw = df_122["X_value"]
    x_122_min = pulses_exp_122_raw.min()
    x_122_max = pulses_exp_122_raw.max()
    raw_pulses_exp_122 = (pulses_exp_122_raw - x_122_min) / (x_122_max - x_122_min) * 33040.0
    cond_exp_122_ms_raw = df_122["Y_value"].copy()

    base_params = default_device(data_dir=data_dir)
    if USE_RAW_EMPIRICAL_RANGE:
        print("Using empirical physical conductance directly from df['Y_value'] (in mS) matching EmpiricalComparison.")
        raw_cond_exp_ns_ms = cond_exp_ns_ms_raw.values if hasattr(cond_exp_ns_ms_raw, 'values') else np.asarray(cond_exp_ns_ms_raw)
        raw_cond_exp_sat_ms = cond_exp_sat_ms_raw.values if hasattr(cond_exp_sat_ms_raw, 'values') else np.asarray(cond_exp_sat_ms_raw)
        raw_cond_exp_122_ms = cond_exp_122_ms_raw.values if hasattr(cond_exp_122_ms_raw, 'values') else np.asarray(cond_exp_122_ms_raw)
    else:
        # Scale empirical data to match the default model ranges
        ns_range, sat_range, range_122 = get_default_ranges(data_dir=data_dir)
        
        c_ns_raw_min = cond_exp_ns_ms_raw.min()
        c_ns_raw_max = cond_exp_ns_ms_raw.max()
        raw_cond_exp_ns_ms = ns_range[0] + ((cond_exp_ns_ms_raw - c_ns_raw_min) / (c_ns_raw_max - c_ns_raw_min)) * (ns_range[1] - ns_range[0])
        
        c_sat_raw_min = cond_exp_sat_ms_raw.min()
        c_sat_raw_max = cond_exp_sat_ms_raw.max()
        raw_cond_exp_sat_ms = sat_range[0] + ((cond_exp_sat_ms_raw - c_sat_raw_min) / (c_sat_raw_max - c_sat_raw_min)) * (sat_range[1] - sat_range[0])
        
        c_122_raw_min = cond_exp_122_ms_raw.min()
        c_122_raw_max = cond_exp_122_ms_raw.max()
        raw_cond_exp_122_ms = range_122[0] + ((cond_exp_122_ms_raw - c_122_raw_min) / (c_122_raw_max - c_122_raw_min)) * (range_122[1] - range_122[0])

    # Resample monotonic PCHIP target curves into 1,000 dense points matching FittedFigures
    pulses_exp_ns, cond_exp_ns_ms = create_pchip_dense_dataset(raw_pulses_exp_ns, raw_cond_exp_ns_ms, num_points=1000)
    pulses_exp_sat, cond_exp_sat_ms = create_pchip_dense_dataset(raw_pulses_exp_sat, raw_cond_exp_sat_ms, num_points=1000)
    pulses_exp_122, cond_exp_122_ms = create_pchip_dense_dataset(raw_pulses_exp_122, raw_cond_exp_122_ms, num_points=1000)
    
    # Save target versus raw digitized comparison plots
    plots_dir = os.path.join(fitting_dir, "TargetVsRawPlots")
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # Non-saturation: Raw Digitized vs. Scaled/Mapped Target
    axes[0, 0].scatter(df_ns["X_value"], df_ns["Y_value"], color="#d9b310", alpha=0.6, edgecolors="k", s=25)
    axes[0, 0].set_title("Non-Saturating Raw Digitized Data", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Raw X (Digitized Units)", fontsize=10)
    axes[0, 0].set_ylabel("Raw Y (Digitized % / Current)", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    axes[0, 1].plot(pulses_exp_ns, cond_exp_ns_ms, color="#328cc1", linewidth=2, label="1000-pt PCHIP Target")
    axes[0, 1].scatter(raw_pulses_exp_ns, raw_cond_exp_ns_ms, color="#d9b310", alpha=0.6, edgecolors="k", s=25, label="Raw Mapped")
    axes[0, 1].set_title("Non-Saturating Target Mapped Data (1000-pt PCHIP)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Pulse Count (n)", fontsize=10)
    axes[0, 1].set_ylabel("Conductance (mS)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(frameon=True, fontsize=9)
    
    # Saturation: Raw Digitized vs. Scaled/Mapped Target
    axes[1, 0].scatter(df_sat["X_value"], df_sat["Y_value"], color="#ff5a5f", alpha=0.6, edgecolors="k", s=25)
    axes[1, 0].set_title("Saturating Raw Digitized Data", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Raw X (Digitized Units)", fontsize=10)
    axes[1, 0].set_ylabel("Raw Y (Digitized % / Current)", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    axes[1, 1].plot(pulses_exp_sat, cond_exp_sat_ms, color="#0b3c5d", linewidth=2, label="1000-pt PCHIP Target")
    axes[1, 1].scatter(raw_pulses_exp_sat, raw_cond_exp_sat_ms, color="#ff5a5f", alpha=0.6, edgecolors="k", s=25, label="Raw Mapped")
    axes[1, 1].set_title("Saturating Target Mapped Data (1000-pt PCHIP)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Pulse Count (n)", fontsize=10)
    axes[1, 1].set_ylabel("Conductance (mS)", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(frameon=True, fontsize=9)
    
    # 1.22V Saturation: Raw Digitized vs. Scaled/Mapped Target
    axes[2, 0].scatter(df_122["X_value"], df_122["Y_value"], color="#10b981", alpha=0.6, edgecolors="k", s=25)
    axes[2, 0].set_title("1.22V Saturating Raw Digitized Data", fontsize=11, fontweight="bold")
    axes[2, 0].set_xlabel("Raw X (Digitized Units)", fontsize=10)
    axes[2, 0].set_ylabel("Raw Y (Digitized % / Current)", fontsize=10)
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    
    axes[2, 1].plot(pulses_exp_122, cond_exp_122_ms, color="#1e3a8a", linewidth=2, label="1000-pt PCHIP Target")
    axes[2, 1].scatter(raw_pulses_exp_122, raw_cond_exp_122_ms, color="#10b981", alpha=0.6, edgecolors="k", s=25, label="Raw Mapped")
    axes[2, 1].set_title("1.22V Saturating Target Mapped Data (1000-pt PCHIP)", fontsize=11, fontweight="bold")
    axes[2, 1].set_xlabel("Pulse Count (n)", fontsize=10)
    axes[2, 1].set_ylabel("Conductance (mS)", fontsize=10)
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].legend(frameon=True, fontsize=9)
    
    plt.suptitle(f"Raw vs. Transformed Target Data - {device_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    target_vs_raw_plot_path = os.path.join(plots_dir, f"{device_name}_target_vs_raw_subset_{selection_str}.png")
    plt.savefig(target_vs_raw_plot_path, dpi=300)
    plt.close()
    print(f"Saved target vs raw digitized comparison plot to: {target_vs_raw_plot_path}")
    
    # base_params is loaded above
    
    t_period = 160e-9
    t_width_pot = 80e-9
    t_width_dep = 60e-9
    vpot = PulseSource(node=None, amp=0.9, period=t_period, width=t_width_pot)
    vdep = PulseSource(node=None, amp=-0.75, period=t_period, width=t_width_dep)
    
    # Initialize the global variables in the main process (used by L-BFGS-B & plots)
    tol_shared = multiprocessing.Value('d', 3e-3) # Start with coarse DE tolerance
    best_cost_shared = multiprocessing.Value('d', 1e9) # Shared best cost value
    print("Setting coarse variable time-step tolerance (tol=3e-3) for Differential Evolution...", flush=True)
    init_worker(
        raw_cond_exp_ns_ms, raw_pulses_exp_ns, 
        raw_cond_exp_sat_ms, raw_pulses_exp_sat,
        raw_cond_exp_122_ms, raw_pulses_exp_122,
        base_params, t_period, t_width_pot, vpot, vdep,
        tol_shared, best_cost_shared,
        fit_ns, fit_sat, fit_122
    )
    
    w_sat_info = compute_nonlinearity_weights(pulses_exp_sat, cond_exp_sat_ms, w_base=0.50, raw_x=raw_pulses_exp_sat, raw_y=raw_cond_exp_sat_ms)
    peak_idx = int(np.argmax(cond_exp_sat_ms))
    collapse_idx = min(peak_idx + 1, len(w_sat_info) - 1)
    print(f"Dynamic nonlinearity weighting enabled: Peak weight = {w_sat_info[peak_idx]:.2f}, collapse point weight = {w_sat_info[collapse_idx]:.2f}, baseline tail weight = {np.min(w_sat_info):.2f}")
    
    # Optimization Bounds (Widened for robustness, thermal limits physically constrained)
    bounds = [
        (1e6, 3e7),          # 0: kappa
        (0.1, 10.0),         # 1: alpha
        (0.38, 0.85),        # 2: vTh (widened from 0.50 to 0.38 to accommodate Ru_azo baseline and relieve boundary pinning)
        (5.2, 6.9),          # 3: gScale (physically constrained to Ru_azo 7.13 mS transconductance)
        (0.001, 0.05),       # 4: rC (capped to prevent unphysical high-gScale compression)
        (0.001, 0.30),       # 5: vSmooth
        (150.0, 1500.0),     # 6: nSmooth (floor raised from 5.0 to 150.0 to prevent abrupt saturation overshoot)
        (0.1, 150.0),        # 7: kDischarge (apex rapid drop boost amplitude)
        (0.15, 0.60),        # 8: E_a (eV) (physically capped for Ru-azo isomerization)
        (100.0, 10000),      # 9: R_th (K/W) (capped to prevent unphysical thermal explosion)
        (1e-9, 1e-7),        # 10: tau_th (s) (restricted to prevent numerical stiffness)
        (-0.9, 0),           # 11: gamma (dynamic tunneling barrier)
        (0.0, 5.0),          # 12: beta_alpha (voltage rate exponent coupling)
        (-5.0, 10.0),        # 13: beta_s (voltage nSmooth clamping coupling, widened to 10.0 for 1.22V shoulder)
        (0.05, 3.0),         # 14: rate_asymmetry (widened from 0.5 to 0.05)
        (0.10, 0.70),        # 15: fDischarge (Option A relaxation curvature: p_decay = 1/fDischarge, 0.10 to 0.70)
        (0.0, 0.95),         # 16: rLatency (shelf latency throttle factor: 0 = unsuppressed, 0.95 = 95% suppressed on shelf)
        (0.10, 1.00)         # 17: rBottomBlend (saturation recovery base landing boundary scalar: 0.10 to 1.00)
    ]
    
    # Map to log/linear optimizer bounds
    optimizer_bounds = []
    for idx, (lo, hi) in enumerate(bounds):
        if LOG_SCALE_PARAMS[idx]:
            optimizer_bounds.append((np.log10(lo), np.log10(hi)))
        else:
            optimizer_bounds.append((lo, hi))
            
    # Tuple of datasets to initialize worker processes with (using compact raw arrays to prevent Windows pipe buffer overflow)
    init_args = (
        raw_cond_exp_ns_ms, raw_pulses_exp_ns, 
        raw_cond_exp_sat_ms, raw_pulses_exp_sat,
        raw_cond_exp_122_ms, raw_pulses_exp_122,
        base_params, t_period, t_width_pot, vpot, vdep,
        tol_shared, best_cost_shared,
        fit_ns, fit_sat, fit_122
    )
    
    from multiprocessing import Pool
    
    if ENABLE_PREWARMING_SWEEP:
        print("\nRunning initial parameter range optimization (ROI boundary shrinking sweep)...", flush=True)
        M = CONDITIONING_SWEEP_SIZE
        x_samples = np.zeros((M, len(optimizer_bounds)))
        for col_idx, (lo, hi) in enumerate(optimizer_bounds):
            # Generate M evenly spaced intervals in [0, 1] for LHS
            intervals = np.linspace(0.0, 1.0, M + 1)
            bin_mins = intervals[:-1]
            bin_maxs = intervals[1:]
            # Sample one random point within each interval
            points_lhs = bin_mins + np.random.uniform(0.0, 1.0, size=M) * (bin_maxs - bin_mins)
            # Shuffle points to break parameter monotonicity
            np.random.shuffle(points_lhs)
            
            # Generate standard uniform random points in [0, 1]
            points_rand = np.random.uniform(0.0, 1.0, size=M)
            
            # Blend coordinates (LHS / standard random) to pull values away from boundaries
            points = PROFILE_LHS_BLEND * points_lhs + (1.0 - PROFILE_LHS_BLEND) * points_rand
            
            x_samples[:, col_idx] = lo + points * (hi - lo)
            
        with Pool(processes=WORKERS, initializer=init_worker, initargs=init_args) as sweep_pool:
            costs = []
            best_cost = np.inf
            # Use chunksize=1 to enable dynamic load-balancing (work-stealing at the process queue level)
            chunksize = 1
            for idx, cost in enumerate(sweep_pool.imap(cost_function, x_samples, chunksize=chunksize)):
                costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                if idx % 10 == 0 or idx == M - 1:
                    update_sweep_dashboard("Random Sweep", idx + 1, M, best_cost)
            clear_dashboard()
            costs = np.array(costs)
        
        # 1. Extract the single best candidate from the initial random sweep
        best_idx = np.argmin(costs)
        cost_best = costs[best_idx]
        x_best = x_samples[best_idx]
        print(f"Best candidate found in initial random sweep: Cost = {cost_best:.6f}")
        
        # Calculate correlation matrix for the top 10% performing candidates to detect parameter coupling
        cutoff = np.percentile(costs, 10)
        good_indices = np.where(costs <= max(cutoff, 30.0))[0]
        if len(good_indices) < 20:
            good_indices = np.argsort(costs)[:100]
        good_samples = x_samples[good_indices]
        
        corr_matrix = np.corrcoef(good_samples.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 2. Construct coordinate-wise profile sweep vectors
        # We will sample K points along each parameter axis
        K = PROFILE_SWEEP_STEPS
        n_params = len(optimizer_bounds)
        total_profile_samples = n_params * K
        x_profile_samples = np.zeros((total_profile_samples, n_params))
        
        # Fill in base values first (locked at x_best)
        for i in range(total_profile_samples):
            x_profile_samples[i, :] = x_best.copy()
            
        # Generate 1D grid for each parameter and place in sweep array
        grids = []
        for col_idx, (lo, hi) in enumerate(optimizer_bounds):
            grid = np.linspace(lo, hi, K)
            grids.append(grid)
            start_row = col_idx * K
            x_profile_samples[start_row : start_row + K, col_idx] = grid
            
        # 3. Evaluate the profile sweep candidates in parallel
        print(f"Running coordinate-wise Profile Likelihood sweeps ({total_profile_samples} total evaluations)...", flush=True)
        with Pool(processes=WORKERS, initializer=init_worker, initargs=init_args) as profile_pool:
            profile_costs = []
            best_profile_cost = np.inf
            # Use chunksize_profile = 1 to enable dynamic load-balancing (work-stealing at the process queue level)
            chunksize_profile = 1
            for idx, cost in enumerate(profile_pool.imap(cost_function, x_profile_samples, chunksize=chunksize_profile)):
                profile_costs.append(cost)
                if cost < best_profile_cost:
                    best_profile_cost = cost
                if idx % 10 == 0 or idx == total_profile_samples - 1:
                    update_sweep_dashboard("Profile Sweeps", idx + 1, total_profile_samples, best_profile_cost)
            clear_dashboard()
            profile_costs = np.array(profile_costs)
        
        # 4. Analyze sweep costs and contract boundaries to the low-cost window
        new_opt_bounds = []
        compaction_ratios = []
        cost_threshold = max(PROFILE_COST_THRESHOLD, 1.5 * cost_best)
        print(f"\nEvaluating Profile Likelihood (Cost Threshold: {cost_threshold:.2f})")
        print(f"  {'Parameter':<12} | {'Original Min':<12} | {'Original Max':<12} | {'Contracted Min':<12} | {'Contracted Max':<12} | {'Max Corr':<8} | {'Comp. Ratio':<11}")
        print("-" * 111)
        
        for col_idx, (lo, hi) in enumerate(optimizer_bounds):
            param_name = PARAM_NAMES[col_idx]
            start_row = col_idx * K
            col_costs = profile_costs[start_row : start_row + K]
            col_grid = grids[col_idx]
            
            # Extract maximum absolute correlation with any OTHER parameter to scale margin
            temp_corr = np.abs(corr_matrix[col_idx, :]).copy()
            temp_corr[col_idx] = 0.0
            max_corr = np.max(temp_corr)
            
            # Scale safety margin from 1.0x (uncorrelated) to 3.0x (highly coupled)
            corr_factor = 1.0 + 2.0 * max_corr
            
            # Find points where cost meets threshold
            valid_indices = np.where(col_costs <= cost_threshold)[0]
            
            if len(valid_indices) >= 2:
                # Set contracted bounds to the min/max grid values that pass the threshold
                p_min = col_grid[valid_indices[0]]
                p_max = col_grid[valid_indices[-1]]
                
                # Apply safety margin scaled dynamically by parameter correlation
                range_width = p_max - p_min
                margin = PROFILE_SAFETY_MARGIN * range_width * corr_factor if range_width > 0.0 else (0.5 * PROFILE_SAFETY_MARGIN) * (hi - lo) * corr_factor
                p_min = max(lo, p_min - margin)
                p_max = min(hi, p_max + margin)
            else:
                # Fallback if no points pass threshold: retain the original unconstrained boundaries
                p_min = lo
                p_max = hi
                
            # Guarantee a minimum search width of 5% of original range
            min_width = 0.05 * (hi - lo)
            if (p_max - p_min) < min_width:
                center = (p_min + p_max) / 2
                p_min = max(lo, center - min_width / 2)
                p_max = min(hi, center + min_width / 2)
                
            new_opt_bounds.append((p_min, p_max))
            
            # Calculate compaction ratio in optimizer (log10 or linear search) space
            comp_ratio = (p_max - p_min) / (hi - lo) if (hi - lo) > 1e-12 else 1.0
            compaction_ratios.append(comp_ratio)
            comp_pct = comp_ratio * 100.0
            
            # Display physical bounds for readability
            lo_phys = lo if not LOG_SCALE_PARAMS[col_idx] else 10**lo
            hi_phys = hi if not LOG_SCALE_PARAMS[col_idx] else 10**hi
            p_min_phys = p_min if not LOG_SCALE_PARAMS[col_idx] else 10**p_min
            p_max_phys = p_max if not LOG_SCALE_PARAMS[col_idx] else 10**p_max
            
            print(f"  {param_name:<12} | {lo_phys:<12.4e} | {hi_phys:<12.4e} | {p_min_phys:<12.4e} | {p_max_phys:<12.4e} | {max_corr:<8.2f} | {comp_pct:<10.1f}%")
            
        optimizer_bounds = new_opt_bounds
        total_volume_compaction = np.prod(compaction_ratios)
        print("-" * 111)
        print(f"Search boundaries successfully contracted (Correlation-Aware).")
        print(f"Total Parameter Search Space Volume contracted to {total_volume_compaction:.2e} of original volume.")
        print("-" * 111 + "\n")
    else:
        print("\nPre-warming ROI boundary contraction is DISABLED.", flush=True)
        print("Differential Evolution will explore the full unconstrained parameter bounds directly.\n", flush=True)
    
    # Generate an initial population seeded with the default parameters
    n_params = len(bounds)
    pop_size_total = POP_SIZE * n_params
    
    init_pop = np.zeros((pop_size_total, n_params))
    for col_idx, (lo, hi) in enumerate(optimizer_bounds):
        # Generate N evenly spaced intervals in [0, 1] for LHS
        intervals = np.linspace(0.0, 1.0, pop_size_total + 1)
        bin_mins = intervals[:-1]
        bin_maxs = intervals[1:]
        # Sample one random point within each interval
        points_lhs = bin_mins + np.random.uniform(0.0, 1.0, size=pop_size_total) * (bin_maxs - bin_mins)
        # Shuffle points to break parameter monotonicity
        np.random.shuffle(points_lhs)
        
        # Generate standard uniform random points in [0, 1]
        points_rand = np.random.uniform(0.0, 1.0, size=pop_size_total)
        
        # Blend coordinates (LHS / standard random) to pull extreme values away from boundaries
        points = DE_LHS_BLEND * points_lhs + (1.0 - DE_LHS_BLEND) * points_rand
        
        # Map to optimizer bounds [lo, hi]
        init_pop[:, col_idx] = lo + points * (hi - lo)
        
    # Map the current default parameters to the first member
    default_p_vals = [
        base_params["kappa"],
        base_params["alpha"],
        base_params["vTh"],
        base_params["gScale"],
        base_params["rC"],
        base_params["vSmooth"],
        max(150.0, base_params["nSmooth"]),
        min(50.0, max(0.1, base_params.get("kDischarge", 5.0))),
        base_params.get("E_a", 0.0),
        base_params.get("R_th", 0.0),
        base_params.get("tau_th", 1e-9),
        base_params.get("gamma", 0.0),
        base_params.get("beta_alpha", 0.0),
        base_params.get("beta_s", 0.0),
        base_params.get("rate_asymmetry", 1.0),
        base_params.get("fDischarge", 0.35),
        base_params.get("rLatency", 0.70),
        base_params.get("rBottomBlend", 1.00)
    ]
    
    # Clamp defaults to physical bounds first to prevent log10(0.0) on uninitialized defaults (e.g. R_th = 0)
    clamped_p_vals = [np.clip(default_p_vals[i], bounds[i][0], bounds[i][1]) for i in range(n_params)]
    default_opt_vals = to_optimizer_params(clamped_p_vals)
    
    # Overwrite the first population member
    for col_idx, (lo, hi) in enumerate(optimizer_bounds):
        init_pop[0, col_idx] = default_opt_vals[col_idx]
        
    txt_path = os.path.join(fitting_dir, f"{device_name}_preset_devices_py_subset_{selection_str}.txt")
    
    best_written_cost = [1e9]
    try:
        print("  [Seed Baseline] Evaluating initial seed candidate on Host...", flush=True)
        initial_seed_cost = cost_function(default_opt_vals)
        best_written_cost[0] = initial_seed_cost
        print(f"  [Seed Baseline] Seed candidate initial evaluation MSE: {initial_seed_cost:.6f}", flush=True)
    except Exception as e:
        print(f"  [Seed Baseline Warning] Failed to evaluate seed candidate on Host: {e}", flush=True)

    figures_dir = os.path.join(fitting_dir, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    def generate_fit_comparison_plot(cand_x, tag="latest"):
        try:
            cand_params = get_dict_params(cand_x, base_params)
            active_sweeps = []
            if fit_ns: active_sweeps.append("NS")
            if fit_sat: active_sweeps.append("SAT")
            if fit_122: active_sweeps.append("122V")
            n_cols = len(active_sweeps)
            if n_cols == 0:
                return

            fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 12))
            ax_cond_list = [axes[0]] if n_cols == 1 else [axes[0, i] for i in range(n_cols)]
            ax_temp_list = [axes[1]] if n_cols == 1 else [axes[1, i] for i in range(n_cols)]
            curr_col = 0

            if fit_ns:
                ax1 = ax_cond_list[curr_col]
                ax4 = ax_temp_list[curr_col]
                curr_col += 1
                sim_ns = MolmemSimulator(device='cpu', backend='numba')
                sim_ns.detailedPrint = False
                candidate_ns = cand_params.copy()
                candidate_ns["nInit"] = 0.0
                sim_ns.addCrossbarMatrix(device_type=candidate_ns, multiThread=False, rows=1, cols=1)
                sim_ns.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
                sim_ns.devices[0]['dev']._forceUpdateConductanceParams()
                max_p_ns = np.max(pulses_exp_ns)
                t_switch_ns = (max_p_ns / 2.0) * t_period
                t_stop_ns = max_p_ns * t_period
                def v_pulse_train_ns(t, sources):
                    if isinstance(t, np.ndarray):
                        out = np.zeros_like(t)
                        mask = t < t_switch_ns
                        if np.any(mask): out[mask] = sources[0].getVoltage(t[mask])
                        if np.any(~mask): out[~mask] = sources[1].getVoltage(t[~mask])
                        return out
                    return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
                edge_points_ns = [t_switch_ns]
                sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=edge_points_ns)
                sim_ns.addDcSource(col=1, voltage=0.0)
                sim_ns.run(tEnd=t_stop_ns, min_dt=1e-11, tol=1e-3, title=None)
                g_sim_best_ns = sim_ns.getConductance(row=1, col=1) * 1e3
                t_arr_ns = sim_ns.getTime()
                sim_pulses_ns = t_arr_ns / t_period
                T_sim_best_ns = sim_ns.getTemperature(row=1, col=1)

                ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax1.scatter(raw_pulses_exp_ns, raw_cond_exp_ns_ms, color="#d9b310", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
                ax1.plot(pulses_exp_ns, cond_exp_ns_ms, color="#d9b310", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
                pulses_dense_ns_arr = np.arange(1, max_p_ns + 1)
                g_sim_best_ns_read = np.interp(pulses_dense_ns_arr - 0.05, sim_pulses_ns, g_sim_best_ns)
                ax1.plot(pulses_dense_ns_arr, g_sim_best_ns_read, color="#328cc1", linewidth=3, label="Fitted Simulation Model")
                ax1.set_title(f"Non-Saturating Sweep ({device_name}) [{tag}]", fontsize=11, fontweight="bold")
                ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax1.set_ylabel("Conductance (mS)", fontsize=10)
                ax1.legend(frameon=True, fontsize=9)

                ax4.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax4.plot(np.linspace(1, max_p_ns, len(T_sim_best_ns)), T_sim_best_ns, color="#d9b310", linewidth=2.5, label="Local Temp")
                ax4.set_title("Non-Saturating Sweep Local Temp", fontsize=11, fontweight="bold")
                ax4.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax4.set_ylabel("Temperature (K)", fontsize=10)
                ax4.legend(frameon=True, fontsize=9)

            if fit_sat:
                ax2 = ax_cond_list[curr_col]
                ax5 = ax_temp_list[curr_col]
                curr_col += 1
                sim_sat = MolmemSimulator(device='cpu', backend='numba')
                sim_sat.detailedPrint = False
                candidate_sat = cand_params.copy()
                candidate_sat["nInit"] = 0.0
                sim_sat.addCrossbarMatrix(device_type=candidate_sat, multiThread=False, rows=1, cols=1)
                sim_sat.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
                sim_sat.devices[0]['dev']._forceUpdateConductanceParams()
                max_p_sat = np.max(pulses_exp_sat)
                t_switch_sat = (max_p_sat / 2.0) * t_period
                t_stop_sat = max_p_sat * t_period
                def v_pulse_train_sat(t, sources):
                    if isinstance(t, np.ndarray):
                        out = np.zeros_like(t)
                        mask = t < t_switch_sat
                        if np.any(mask): out[mask] = sources[0].getVoltage(t[mask])
                        if np.any(~mask): out[~mask] = sources[1].getVoltage(t[~mask])
                        return out
                    return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
                edge_points_sat = [t_switch_sat]
                sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=edge_points_sat)
                sim_sat.addDcSource(col=1, voltage=0.0)
                sim_sat.run(tEnd=t_stop_sat, min_dt=1e-11, tol=1e-3, title=None)
                g_sim_best_sat = sim_sat.getConductance(row=1, col=1) * 1e3
                t_arr_sat = sim_sat.getTime()
                sim_pulses_sat = t_arr_sat / t_period
                T_sim_best_sat = sim_sat.getTemperature(row=1, col=1)

                ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax2.scatter(raw_pulses_exp_sat, raw_cond_exp_sat_ms, color="#ff5a5f", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
                ax2.plot(pulses_exp_sat, cond_exp_sat_ms, color="#ff5a5f", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
                pulses_dense_sat_arr = np.arange(1, max_p_sat + 1)
                g_sim_best_sat_read = np.interp(pulses_dense_sat_arr - 0.05, sim_pulses_sat, g_sim_best_sat)
                ax2.plot(pulses_dense_sat_arr, g_sim_best_sat_read, color="#0b3c5d", linewidth=3, label="Fitted Simulation Model")
                ax2.set_title(f"Saturating Sweep ({device_name}) [{tag}]", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax2.set_ylabel("Conductance (mS)", fontsize=10)
                ax2.legend(frameon=True, fontsize=9)

                ax5.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax5.plot(np.linspace(1, max_p_sat, len(T_sim_best_sat)), T_sim_best_sat, color="#ff5a5f", linewidth=2.5, label="Local Temp")
                ax5.set_title("Saturating Sweep Local Temp", fontsize=11, fontweight="bold")
                ax5.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax5.set_ylabel("Temperature (K)", fontsize=10)
                ax5.legend(frameon=True, fontsize=9)

            if fit_122:
                ax3 = ax_cond_list[curr_col]
                ax6 = ax_temp_list[curr_col]
                curr_col += 1
                sim_122 = MolmemSimulator(device='cpu', backend='numba')
                sim_122.detailedPrint = False
                candidate_122 = cand_params.copy()
                candidate_122["nInit"] = 0.0
                sim_122.addCrossbarMatrix(device_type=candidate_122, multiThread=False, rows=1, cols=1)
                sim_122.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
                sim_122.devices[0]['dev']._forceUpdateConductanceParams()
                max_p_122 = np.max(pulses_exp_122)
                t_switch_122 = (max_p_122 / 2.0) * t_period
                t_stop_122 = max_p_122 * t_period
                vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
                vdep_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
                def v_pulse_train_122(t, sources):
                    if isinstance(t, np.ndarray):
                        out = np.zeros_like(t)
                        mask = t < t_switch_122
                        if np.any(mask): out[mask] = sources[0].getVoltage(t[mask])
                        if np.any(~mask): out[~mask] = sources[1].getVoltage(t[~mask])
                        return out
                    return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
                edge_points_122 = [t_switch_122]
                sim_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep_122], edge_points=edge_points_122)
                sim_122.addDcSource(col=1, voltage=0.0)
                sim_122.run(tEnd=t_stop_122, min_dt=1e-11, tol=1e-3, title=None)
                g_sim_best_122 = sim_122.getConductance(row=1, col=1) * 1e3
                t_arr_122 = sim_122.getTime()
                sim_pulses_122 = t_arr_122 / t_period
                T_sim_best_122 = sim_122.getTemperature(row=1, col=1)

                ax3.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax3.scatter(raw_pulses_exp_122, raw_cond_exp_122_ms, color="#10b981", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
                ax3.plot(pulses_exp_122, cond_exp_122_ms, color="#10b981", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
                pulses_dense_122_arr = np.arange(1, max_p_122 + 1)
                g_sim_best_122_read = np.interp(pulses_dense_122_arr - 0.05, sim_pulses_122, g_sim_best_122)
                ax3.plot(pulses_dense_122_arr, g_sim_best_122_read, color="#1e3a8a", linewidth=3, label="Fitted Simulation Model")
                ax3.set_title(f"1.22V Saturating Sweep ({device_name}) [{tag}]", fontsize=11, fontweight="bold")
                ax3.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax3.set_ylabel("Conductance (mS)", fontsize=10)
                ax3.legend(frameon=True, fontsize=9)

                ax6.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax6.plot(np.linspace(1, max_p_122, len(T_sim_best_122)), T_sim_best_122, color="#10b981", linewidth=2.5, label="Local Temp")
                ax6.set_title("1.22V Saturating Sweep Local Temp", fontsize=11, fontweight="bold")
                ax6.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax6.set_ylabel("Temperature (K)", fontsize=10)
                ax6.legend(frameon=True, fontsize=9)

            plt.tight_layout()
            out_file = os.path.join(figures_dir, f"{device_name}_fit_comparison_subset_{selection_str}_{tag}.png")
            plt.savefig(out_file, dpi=200)
            plt.close(fig)
            print_log(f"  [Live Plot] Saved updated fitting comparison plot to {os.path.basename(out_file)}")
        except Exception as pe:
            print_log(f"  [Live Plot Notice] Could not render live plot: {pe}")

    def de_callback(xk, nit=None, *args, **kwargs):
        cost = cost_function(xk)
        if cost < best_written_cost[0]:
            best_written_cost[0] = cost
            save_preset_txt(xk, base_params, txt_path, func_name, device_name, data_dir)
            print_log(f"  [Incremental Save] Updated best parameters to {os.path.basename(txt_path)} (Cost: {cost:.6f})")
            generate_fit_comparison_plot(xk, tag="latest")
        if nit is not None:
            generate_fit_comparison_plot(xk, tag=f"gen_{nit:03d}")

    # Render baseline comparison plot for the initial seed model
    generate_fit_comparison_plot(default_opt_vals, tag="initial_baseline")
    generate_fit_comparison_plot(default_opt_vals, tag="latest")
            
    print("\nRunning Differential Evolution (Seeded & Mutation-Decaying Global Optimization)...")
    
    with Pool(processes=WORKERS, initializer=init_worker, initargs=init_args) as pool:
        solver = DecayingDESolver(
            cost_function,
            bounds=optimizer_bounds,
            args=(),  # Avoid passing datasets dynamically via args to eliminate IPC serialization overhead
            popsize=POP_SIZE,
            maxiter=MAX_ITER,
            tol=TOL,
            disp=False,
            workers=pool.map,
            updating='deferred',
            init=init_pop,
            mutation=(0.01, 1.0),
            polish=False,
            callback=de_callback,
            pool=pool
        )
        res_de = solver.solve()
        
        print("\nDifferential Evolution finished. Resulting parameters:")
        print(to_physical_params(res_de.x))
        
        print("\nExtracting top unique candidates from Differential Evolution population...")
        # Map normalized population back to optimizer space
        unnormalized_population = np.zeros_like(solver.population)
        for col_idx, (lo, hi) in enumerate(optimizer_bounds):
            unnormalized_population[:, col_idx] = lo + solver.population[:, col_idx] * (hi - lo)

        candidates = []
        candidate_energies = []
        seen_norm = []
        
        best_de_cost = res_de.fun
        
        for idx_p, x_cand_norm in enumerate(solver.population):
            cand_cost = solver.population_energies[idx_p]
            
            # Reject candidates with cost worse than 15% of the best overall cost
            if cand_cost > best_de_cost * 1.15:
                continue
                
            # Reject candidates within 15% distance of already selected parameters in normalized space
            is_unique = True
            for s_norm in seen_norm:
                max_diff = np.max(np.abs(x_cand_norm - s_norm))
                if max_diff < 0.15:
                    is_unique = False
                    break
            
            if is_unique:
                candidates.append(unnormalized_population[idx_p])
                candidate_energies.append(cand_cost)
                seen_norm.append(x_cand_norm)
                if len(candidates) >= 5:
                    break
                    
        print(f"Identified {len(candidates)} unique starting basins for local fine-tuning.")
        
        best_overall_x = res_de.x
        best_overall_cost = res_de.fun
        print(f"Differential Evolution Best Candidate Cost: {best_overall_cost:.6f}")
        
        # Switch to tight variable time-step tolerance (tol=5e-4) for high-precision L-BFGS-B gradient minimization
        tol_shared.value = 5e-4
        best_cost_shared.value = 1e9  # Disable pruning during L-BFGS-B to ensure exact gradients
        print("\nTuning variable time-step tolerance (tol=5e-4) and disabling pruning for L-BFGS-B gradient minimization...", flush=True)
        
        best_overall_cost_ref = [best_overall_cost]
        for idx, (x0_cand, start_energy) in enumerate(zip(candidates, candidate_energies)):
            print_log(f"\n[Basin {idx + 1}/{len(candidates)}] Local minimization starting cost: {start_energy:.6f}...")
            tracker = LocalMinimizationTracker(start_energy, best_overall_cost_ref, de_callback, x0_cand, idx + 1, len(candidates))
            try:
                res_local = minimize(
                    cost_function,
                    x0=x0_cand,
                    args=(),
                    bounds=optimizer_bounds,
                    method='L-BFGS-B',
                    jac=make_parallel_jacobian(pool, cost_function, bounds=optimizer_bounds),  # Pass process pool to resolve JIT contention
                    options={'maxiter': LBFGS_MAX_ITER, 'disp': False},
                    callback=tracker
                )
                cand_cost = res_local.fun
                best_x_candidate = res_local.x
            except StopIteration:
                cand_cost = tracker.best_cost
                best_x_candidate = tracker.best_x
                print_log(f"[Basin {idx + 1}/{len(candidates)}] Local minimization stopped early due to no improvement in 3 consecutive iterations.")
            
            print_log(f"[Basin {idx + 1}/{len(candidates)}] Tuning complete. Final cost: {cand_cost:.6f}")
            if cand_cost < best_overall_cost_ref[0]:
                best_overall_cost_ref[0] = cand_cost
                best_overall_x = best_x_candidate
                de_callback(best_overall_x)
        best_overall_cost = best_overall_cost_ref[0]

            
    clear_dashboard()
    best_x = best_overall_x
    best_cost = best_overall_cost
    print(f"\nOptimization Completed. Best cost (MSE) = {best_cost:.6f}")
    
    final_params = get_dict_params(best_x, base_params)
    
    # Save final optimized configuration text file for Devices.py
    print(f"Saving final configuration dictionary to {txt_path}...")
    save_preset_txt(best_x, base_params, txt_path, func_name, device_name, data_dir)
    
    with open(txt_path, "r") as f:
        preset_str = f.read()
        
    print("\n=== OPTIMIZED CONFIGURATION PRESET ===")
    print(preset_str)
    print("======================================\n")
    
    # Create comparison plots for selected sweeps
    print("Generating and saving verification plot using true simulator...")
    
    active_sweeps = []
    if fit_ns: active_sweeps.append("NS")
    if fit_sat: active_sweeps.append("SAT")
    if fit_122: active_sweeps.append("122V")
    
    n_cols = len(active_sweeps)
    if n_cols == 0:
        print("No active sweeps to plot.")
        return
        
    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 12))
    
    if n_cols == 1:
        ax_cond_list = [axes[0]]
        ax_temp_list = [axes[1]]
    else:
        ax_cond_list = [axes[0, i] for i in range(n_cols)]
        ax_temp_list = [axes[1, i] for i in range(n_cols)]
        
    curr_col = 0
    
    if fit_ns:
        ax1 = ax_cond_list[curr_col]
        ax4 = ax_temp_list[curr_col]
        curr_col += 1
        
        sim_ns = MolmemSimulator(device='cpu', backend='numba')
        sim_ns.detailedPrint = False
        candidate_ns = final_params.copy()
        candidate_ns["nInit"] = 0.0
        sim_ns.addCrossbarMatrix(device_type=candidate_ns, multiThread=False, rows=1, cols=1)
        max_p_ns = np.max(pulses_exp_ns)
        t_switch_ns = (max_p_ns / 2.0) * t_period
        t_stop_ns = max_p_ns * t_period
        def v_pulse_train_ns(t, sources):
            if isinstance(t, np.ndarray):
                out = np.zeros_like(t)
                mask = t < t_switch_ns
                if np.any(mask):
                    out[mask] = sources[0].getVoltage(t[mask])
                if np.any(~mask):
                    out[~mask] = sources[1].getVoltage(t[~mask])
                return out
            return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
        pulses_dense_ns = np.linspace(1, max_p_ns, int(max_p_ns))
        edge_points_ns = [t_switch_ns]
        sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=edge_points_ns)
        sim_ns.addDcSource(col=1, voltage=0.0)
        sim_ns.run(tEnd=t_stop_ns, min_dt=1e-11, tol=1e-3, title=None)
        g_sim_best_ns = sim_ns.getConductance(row=1, col=1) * 1e3
        t_arr_ns = sim_ns.getTime()
        sim_pulses_ns = t_arr_ns / t_period
        T_sim_best_ns = sim_ns.getTemperature(row=1, col=1)
        
        ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax1.scatter(raw_pulses_exp_ns, raw_cond_exp_ns_ms, color="#d9b310", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
        ax1.plot(pulses_exp_ns, cond_exp_ns_ms, color="#d9b310", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
        pulses_dense_ns_arr = np.arange(1, max_p_ns + 1)
        g_sim_best_ns_read = np.interp(pulses_dense_ns_arr - 0.05, sim_pulses_ns, g_sim_best_ns)
        ax1.plot(pulses_dense_ns_arr, g_sim_best_ns_read, color="#328cc1", linewidth=3, label="Fitted Simulation Model")
        ax1.set_title(f"Non-Saturating Sweep ({device_name})", fontsize=11, fontweight="bold", pad=15)
        ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
        ax1.set_ylabel("Conductance (mS)", fontsize=10)
        ax1.set_xlim(-0.03 * max_p_ns, 1.05 * max_p_ns)
        ax1.set_ylim(0, max(float(np.max(g_sim_best_ns_read)), float(np.max(cond_exp_ns_ms))) * 1.1)
        half_p_ns = max_p_ns / 2.0
        ticks_ns = [1, half_p_ns, max_p_ns]
        tick_labels_ns = ["1", f"{half_p_ns/1000:.1f}k", f"{max_p_ns/1000:.1f}k"]
        ax1.set_xticks(ticks_ns)
        ax1.set_xticklabels(tick_labels_ns)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
        ax4.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax4.plot(np.linspace(1, max_p_ns, len(T_sim_best_ns)), T_sim_best_ns, color="#d9b310", linewidth=2.5, label="Local Temp")
        ax4.set_title("Non-Saturating Sweep Local Temp", fontsize=11, fontweight="bold", pad=15)
        ax4.set_xlabel("Pulse Counts (n)", fontsize=10)
        ax4.set_ylabel("Temperature (K)", fontsize=10)
        ax4.set_xlim(-0.03 * max_p_ns, 1.05 * max_p_ns)
        ax4.set_xticks(ticks_ns)
        ax4.set_xticklabels(tick_labels_ns)
        ax4.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    if fit_sat:
        ax2 = ax_cond_list[curr_col]
        ax5 = ax_temp_list[curr_col]
        curr_col += 1
        
        sim_sat = MolmemSimulator(device='cpu', backend='numba')
        sim_sat.detailedPrint = False
        candidate_sat = final_params.copy()
        candidate_sat["nInit"] = 0.0
        sim_sat.addCrossbarMatrix(device_type=candidate_sat, multiThread=False, rows=1, cols=1)
        max_p_sat = np.max(pulses_exp_sat)
        t_switch_sat = (max_p_sat / 2.0) * t_period
        t_stop_sat = max_p_sat * t_period
        def v_pulse_train_sat(t, sources):
            if isinstance(t, np.ndarray):
                out = np.zeros_like(t)
                mask = t < t_switch_sat
                if np.any(mask):
                    out[mask] = sources[0].getVoltage(t[mask])
                if np.any(~mask):
                    out[~mask] = sources[1].getVoltage(t[~mask])
                return out
            return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
        pulses_dense_sat = np.linspace(1, max_p_sat, int(max_p_sat))
        edge_points_sat = [t_switch_sat]
        sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=edge_points_sat)
        sim_sat.addDcSource(col=1, voltage=0.0)
        sim_sat.run(tEnd=t_stop_sat, min_dt=1e-11, tol=1e-3, title=None)
        g_sim_best_sat = sim_sat.getConductance(row=1, col=1) * 1e3
        t_arr_sat = sim_sat.getTime()
        sim_pulses_sat = t_arr_sat / t_period
        T_sim_best_sat = sim_sat.getTemperature(row=1, col=1)
        
        ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax2.scatter(raw_pulses_exp_sat, raw_cond_exp_sat_ms, color="#ff5a5f", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
        ax2.plot(pulses_exp_sat, cond_exp_sat_ms, color="#ff5a5f", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
        pulses_dense_sat_arr = np.arange(1, max_p_sat + 1)
        g_sim_best_sat_read = np.interp(pulses_dense_sat_arr - 0.05, sim_pulses_sat, g_sim_best_sat)
        ax2.plot(pulses_dense_sat_arr, g_sim_best_sat_read, color="#0b3c5d", linewidth=3, label="Fitted Simulation Model")
        ax2.set_title(f"Saturating Sweep ({device_name})", fontsize=11, fontweight="bold", pad=15)
        ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
        ax2.set_ylabel("Conductance (mS)", fontsize=10)
        ax2.set_xlim(-0.03 * max_p_sat, 1.05 * max_p_sat)
        ax2.set_ylim(0, max(float(np.max(g_sim_best_sat_read)), float(np.max(cond_exp_sat_ms))) * 1.1)
        half_p_sat = max_p_sat / 2.0
        ticks_sat = [1, half_p_sat, max_p_sat]
        tick_labels_sat = ["1", f"{half_p_sat/1000:.1f}k", f"{max_p_sat/1000:.1f}k"]
        ax2.set_xticks(ticks_sat)
        ax2.set_xticklabels(tick_labels_sat)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
        ax5.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax5.plot(np.linspace(1, max_p_sat, len(T_sim_best_sat)), T_sim_best_sat, color="#ff5a5f", linewidth=2.5, label="Local Temp")
        ax5.set_title("Saturating Sweep Local Temp", fontsize=11, fontweight="bold", pad=15)
        ax5.set_xlabel("Pulse Counts (n)", fontsize=10)
        ax5.set_ylabel("Temperature (K)", fontsize=10)
        ax5.set_xlim(-0.03 * max_p_sat, 1.05 * max_p_sat)
        ax5.set_xticks(ticks_sat)
        ax5.set_xticklabels(tick_labels_sat)
        ax5.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    if fit_122:
        ax3 = ax_cond_list[curr_col]
        ax6 = ax_temp_list[curr_col]
        curr_col += 1
        
        sim_122 = MolmemSimulator(device='cpu', backend='numba')
        sim_122.detailedPrint = False
        candidate_122 = final_params.copy()
        candidate_122["nInit"] = 0.0
        sim_122.addCrossbarMatrix(device_type=candidate_122, multiThread=False, rows=1, cols=1)
        max_p_122 = np.max(pulses_exp_122)
        t_switch_122 = (max_p_122 / 2.0) * t_period
        t_stop_122 = max_p_122 * t_period
        vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
        vdep_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
        def v_pulse_train_122(t, sources):
            if isinstance(t, np.ndarray):
                out = np.zeros_like(t)
                mask = t < t_switch_122
                if np.any(mask):
                    out[mask] = sources[0].getVoltage(t[mask])
                if np.any(~mask):
                    out[~mask] = sources[1].getVoltage(t[~mask])
                return out
            return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
        pulses_dense_122 = np.linspace(1, max_p_122, int(max_p_122))
        edge_points_122 = [t_switch_122]
        sim_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep_122], edge_points=edge_points_122)
        sim_122.addDcSource(col=1, voltage=0.0)
        sim_122.run(tEnd=t_stop_122, min_dt=1e-11, tol=1e-3, title=None)
        g_sim_best_122 = sim_122.getConductance(row=1, col=1) * 1e3
        t_arr_122 = sim_122.getTime()
        sim_pulses_122 = t_arr_122 / t_period
        T_sim_best_122 = sim_122.getTemperature(row=1, col=1)
        
        ax3.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax3.scatter(raw_pulses_exp_122, raw_cond_exp_122_ms, color="#10b981", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
        ax3.plot(pulses_exp_122, cond_exp_122_ms, color="#10b981", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
        pulses_dense_122_arr = np.arange(1, max_p_122 + 1)
        g_sim_best_122_read = np.interp(pulses_dense_122_arr - 0.05, sim_pulses_122, g_sim_best_122)
        ax3.plot(pulses_dense_122_arr, g_sim_best_122_read, color="#1e3a8a", linewidth=3, label="Fitted Simulation Model")
        ax3.set_title(f"1.22V Saturating Sweep ({device_name})", fontsize=11, fontweight="bold", pad=15)
        ax3.set_xlabel("Pulse Counts (n)", fontsize=10)
        ax3.set_ylabel("Conductance (mS)", fontsize=10)
        ax3.set_xlim(-0.03 * max_p_122, 1.05 * max_p_122)
        ax3.set_ylim(0, max(float(np.max(g_sim_best_122_read)), float(np.max(cond_exp_122_ms))) * 1.1)
        half_p_122 = max_p_122 / 2.0
        ticks_122 = [1, half_p_122, max_p_122]
        tick_labels_122 = ["1", f"{half_p_122/1000:.1f}k", f"{max_p_122/1000:.1f}k"]
        ax3.set_xticks(ticks_122)
        ax3.set_xticklabels(tick_labels_122)
        ax3.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
        ax6.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax6.plot(np.linspace(1, max_p_122, len(T_sim_best_122)), T_sim_best_122, color="#10b981", linewidth=2.5, label="Local Temp")
        ax6.set_title("1.22V Saturating Sweep Local Temp", fontsize=11, fontweight="bold", pad=15)
        ax6.set_xlabel("Pulse Counts (n)", fontsize=10)
        ax6.set_ylabel("Temperature (K)", fontsize=10)
        ax6.set_xlim(-0.03 * max_p_122, 1.05 * max_p_122)
        ax6.set_xticks(ticks_122)
        ax6.set_xticklabels(tick_labels_122)
        ax6.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
    os.makedirs(os.path.join(fitting_dir, "Figures"), exist_ok=True)
    out_plot_path = os.path.join(fitting_dir, "Figures", f"{device_name}_fit_comparison_subset_{selection_str}.png")
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    plt.close()
    
    print(f"Comparison plot successfully saved to {out_plot_path}")
    print("--------------------------------------------------")
    
    if update_registry_after_fit:
        n_init_best = 0.0
        register_or_update_device(device_name, final_params, n_init_best, devices_py_path, exists_in_registry)
        
    print(f"  [{multiprocessing.current_process().name}] Saved Config: kappa={final_params['kappa']:.4e}, alpha={final_params['alpha']:.3f}, vTh={final_params['vTh']:.3f}, vRefPos={final_params['vRefPos']:.3f}, vRefNeg={final_params['vRefNeg']:.3f}, gScale={final_params['gScale']:.3f}, rC={final_params['rC']:.3f}, nSmooth={final_params['nSmooth']:.1f}, kDisch={final_params['kDischarge']:.1f}, E_a={final_params['E_a']:.3f}, R_th={final_params['R_th']:.1f}, tau_th={final_params['tau_th']:.2e}, gamma={final_params.get('gamma', 0.0):.3f}, rLat={final_params.get('rLatency', 0.5):.3f} -> Cost={best_cost:.4f}", flush=True)

if __name__ == "__main__":
    main()
