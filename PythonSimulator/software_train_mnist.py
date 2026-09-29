#!/usr/bin/env python3
"""
===============================================================================
TRUE HARDWARE CROSSBAR SIMULATION: FULL MNIST 10-DIGIT CLASSIFIER (0–9)
===============================================================================
Performs true analog physics-grounded crossbar simulation using MolmemSimulator:
  - Architecture: 64 (Inputs) -> 128 (Hidden) -> 10 (Outputs) [Configurable]
  - Layer 1 Crossbar: 64 rows x 256 columns (128 differential pairs: pos/neg)
  - Layer 2 Crossbar: 128 rows x 20 columns (10 differential pairs: pos/neg)
  - Physics: True analog conductance sensing, non-linear tunneling, contact
             resistance (rC), DAC/ADC 14-bit quantization, and pulsed programming.
===============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import molmem_lib
from molmem_lib import MolmemSimulator, MolmemTiledSimulator, silenceMolmemLib


def shuffle_data(X, Y, seed=None):
    """Native NumPy in-unison array permutation (eliminates scikit-learn dependency)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
    else:
        perm = np.random.permutation(len(X))
    return X[perm], Y[perm]


def softmax(x):
    """Numerically stable softmax for 2D logit matrices."""
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-15)


def encode_differential_weights(W):
    """
    Encodes signed floating-point weights [-1.0, 1.0] into physical differential
    crossbar column pairs [0.0, 1.0]: W_diff = [W_pos, W_neg].
    """
    w_pos = np.clip(np.maximum(W, 0.0), 0.0, 1.0)
    w_neg = np.clip(np.maximum(-W, 0.0), 0.0, 1.0)
    return np.ascontiguousarray(np.hstack([w_pos, w_neg]), dtype=np.float64)


def choose_tile_dim(total_dim, preferred_tile=32):
    """Finds the largest divisor of total_dim <= preferred_tile, or total_dim if none found."""
    for d in range(preferred_tile, 0, -1):
        if total_dim % d == 0:
            return d
    return total_dim


def build_crossbar(instance_name, rows, out_units, device="auto", tile_rows=None, tile_cols=None):
    """
    Instantiates and configures a physical tiled 1T1R memristor macro-crossbar array with
    differential column pairs for signed weights.
    """
    cols = out_units * 2  # Each logical output unit has (+, -) differential columns
    if tile_rows is None:
        tile_rows = choose_tile_dim(rows, 32)
    if tile_cols is None:
        tile_cols = choose_tile_dim(cols, 64)

    sim = MolmemTiledSimulator(instanceName=instance_name, device=device)
    sim.recordHistory = False
    sim.addTiledCrossbarMatrix(
        total_rows=rows,
        total_cols=cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        weightBits=14,
        inputBits=14,
        outputBits=14,
        detailedPrint=False,
        architecture='1T1R',
        max_vWrite=5.0,
        max_pw=320e-9,
        UpdatesPer="Column"
    )
    return sim


def parse_args():
    parser = argparse.ArgumentParser(description="True Hardware Crossbar Training on Full 10-Class MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10, recommended 5-10 for 8x8 MNIST)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Number of hidden units in Layer 1 (default: 256)")
    parser.add_argument("--batch-size", type=int, default=128, help="Base mini-batch size (default: 128)")
    parser.add_argument("--parallel-batches", type=int, default=4, help="Initial parallel batches to evaluate simultaneously (default: 4)")
    parser.add_argument("--target-cpu", type=float, default=80.0, help="Target CPU utilization percent for dynamic batch scaling (default: 80.0, 0 to disable)")
    parser.add_argument("--min-parallel", type=int, default=1, help="Minimum parallel batch multiplier (default: 1)")
    parser.add_argument("--max-parallel", type=int, default=16, help="Maximum parallel batch multiplier (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial peak learning rate (default: 0.0005 with Adam)")
    parser.add_argument("--lr-min", type=float, default=0.00005, help="Minimum annealed learning rate with cosine decay (default: 0.00005)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay regularization (default: 1e-4)")
    parser.add_argument("--gain", type=float, default=500.0, help="Hardware TIA analog output gain scale (default: 500.0)")
    parser.add_argument("--update-freq", type=int, default=8, help="Frequency of physical crossbar weight updates in mini-batches (default: 8)")
    parser.add_argument("--train-samples", type=int, default=60000, help="Max train samples to use (default: 60000)")
    parser.add_argument("--test-samples", type=int, default=10000, help="Max test samples to use (default: 10000)")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device target ('auto', 'gpu', 'cpu')")
    parser.add_argument("--fine-tune", action="store_true", help="Enable closed-loop fine-tuning pulses during weight updates")
    parser.add_argument("--verbose", action="store_true", help="Keep internal molmem_lib diagnostic prints unsilenced")
    return parser.parse_args()


def main():
    args = parse_args()
    hidden_dim = args.hidden_dim

    # Silence internal molmem_lib operational prints for clean testbench console output
    if not args.verbose:
        silenceMolmemLib(True)

    print("==========================================================================")
    print("      MOLMEM PHYSICAL HARDWARE CROSSBAR SIMULATION: FULL MNIST (0–9)      ")
    print("==========================================================================")
    print(f"  Network Topology  : 64 (Inputs) -> {hidden_dim} (Hidden) -> 10 (Digits 0-9)")
    print(f"  Quantization      : 14-bit Weight DAC | 14-bit Input DTC | 14-bit Output ADC")
    print("  Physics Transport : Non-Linear Tunneling + Nodal Contact Resistance (Rc)")
    print(f"  Signal Conditioning: Dynamic TIA Gain ({args.gain}x) + Running Activation Normalization")
    print("==========================================================================\n")

    # 1. Load the full MNIST8 dataset (all 10 digits)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MNIST8_group.npy')
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found.")

    data = np.load(data_path, allow_pickle=True).item()
    X_train_full = data['train_X']
    train_y_full = data['train_y'].flatten()
    X_test_full  = data['test_X']
    test_y_full  = data['test_y'].flatten()

    # Slice to requested sample sizes
    n_train = min(len(X_train_full), args.train_samples)
    n_test  = min(len(X_test_full), args.test_samples)

    X_train = np.ascontiguousarray(X_train_full[:n_train], dtype=np.float64)
    train_y = train_y_full[:n_train]
    Y_train = np.eye(10, dtype=np.float64)[train_y]

    X_test  = np.ascontiguousarray(X_test_full[:n_test], dtype=np.float64)
    test_y  = test_y_full[:n_test]

    X_train, Y_train = shuffle_data(X_train, Y_train, seed=42)

    print(f"[Dataset] Full 10-Class MNIST Loaded: {n_train} Train samples | {n_test} Test samples")
    print(f"[Dataset] Target Classes: {sorted(np.unique(train_y).tolist())}")
    target_cpu_info = f" | Target CPU: {args.target_cpu}% (Dynamic)" if (HAS_PSUTIL and args.target_cpu > 0) else ""
    print(f"[Training] Hidden Units: {hidden_dim} | Base Batch: {args.batch_size} | Parallel Batches: {args.parallel_batches}x{target_cpu_info} | Epochs: {args.epochs} | LR: {args.lr} -> {args.lr_min} (Cosine) | TIA Gain: {args.gain}x | Update Freq: Every {args.update_freq} Batches")

    # 2. Build Physical Tiled Crossbars
    print("\n[Hardware] Allocating and initializing physical tiled memristor crossbars...")
    sim_layer1 = build_crossbar(f"MNIST_Layer1_64x{hidden_dim * 2}", rows=64, out_units=hidden_dim, device=args.device)
    sim_layer2 = build_crossbar(f"MNIST_Layer2_{hidden_dim}x20", rows=hidden_dim, out_units=10, device=args.device)
    print(f"  Layer 1 Tiled Array: 64 Rows x {hidden_dim * 2} Columns ({sim_layer1.num_row_tiles}x{sim_layer1.num_col_tiles} Grid of {sim_layer1.tile_rows}x{sim_layer1.tile_cols} Tiles)")
    print(f"  Layer 2 Tiled Array: {hidden_dim} Rows x 20 Columns ({sim_layer2.num_row_tiles}x{sim_layer2.num_col_tiles} Grid of {sim_layer2.tile_rows}x{sim_layer2.tile_cols} Tiles)")

    # 3. Initialize Mathematical Weights (Xavier/Glorot uniform initialization + zero threshold bias init)
    scale_w1 = np.sqrt(6.0 / (64 + hidden_dim))
    scale_w2 = np.sqrt(6.0 / (hidden_dim + 10))
    W1 = np.random.uniform(-scale_w1, scale_w1, (64, hidden_dim))
    W2 = np.random.uniform(-scale_w2, scale_w2, (hidden_dim, 10))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    b2 = np.zeros(10, dtype=np.float64)

    # Hardware-Aware Adam Optimizer State
    m_W1 = np.zeros_like(W1)
    v_W1 = np.zeros_like(W1)
    m_W2 = np.zeros_like(W2)
    v_W2 = np.zeros_like(W2)
    m_b1 = np.zeros_like(b1)
    v_b1 = np.zeros_like(b1)
    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    opt_step = 0
    running_h_scale = None

    # Initial physical crossbar programming
    print("[Hardware] Programming initial weights into memristor filaments across tiled sub-arrays...")
    sim_layer1.updateCrossbarWeights(encode_differential_weights(W1), detailedPrint=False, fineTune=args.fine_tune, silentUpdate=True)
    sim_layer2.updateCrossbarWeights(encode_differential_weights(W2), detailedPrint=False, fineTune=args.fine_tune, silentUpdate=True)
    print("[Hardware] Tiled crossbars initialized and ready for parallel analog inference!\n")

    # 4. Training Loop with Dynamic Parallel Batching targeting 80% CPU Utilization
    current_parallel = max(args.min_parallel, min(args.max_parallel, args.parallel_batches))
    if HAS_PSUTIL and args.target_cpu > 0:
        psutil.cpu_percent(interval=None)  # Warm up non-blocking CPU baseline

    print(f">>> Starting Physical Tiled Crossbar Training ({args.epochs} Epochs, Dynamic Parallel Super-Batches) <<<")

    for epoch in range(1, args.epochs + 1):
        X_train, Y_train = shuffle_data(X_train, Y_train)
        epoch_start_time = time.perf_counter()
        epoch_loss = 0.0
        epoch_correct = 0
        total_seen = 0
        batches_accumulated = 0.0
        cursor = 0
        step_idx = 0

        # Cosine Annealing Learning Rate Schedule
        progress_epoch = (epoch - 1) / max(1, args.epochs - 1)
        current_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + np.cos(np.pi * progress_epoch))

        while cursor < n_train:
            # Determine dynamic batch slice
            current_bs = min(current_parallel * args.batch_size, n_train - cursor)
            b_start = cursor
            b_end = cursor + current_bs
            X_batch = X_train[b_start:b_end]
            Y_batch = Y_train[b_start:b_end]
            bs = b_end - b_start

            # --- TRUE PHYSICAL FORWARD PASS ---
            # Layer 1: PWM inference through 64x(2*hidden_dim) tiled memristor crossbar
            i1 = sim_layer1.passInput(X_batch, detailedPrint=False, appendHistory=False)
            h_raw = (i1[:, :hidden_dim] - i1[:, hidden_dim:]) + b1  # Differential sensing + threshold bias DAC
            h_act = np.maximum(0.0, h_raw)   # ReLU activation

            # Robust batch-level activation reference (eliminates volatile per-sample division noise)
            h_scale = float(np.percentile(h_act, 95) + 1e-6)
            running_h_scale = h_scale if running_h_scale is None else (0.95 * running_h_scale + 0.05 * h_scale)
            h_norm = np.clip(h_act / h_scale, 0.0, 1.0)  # Normalized to DAC range [0, 1]

            # Layer 2: PWM inference through hidden_dim x 20 tiled memristor crossbar
            i2 = sim_layer2.passInput(h_norm, detailedPrint=False, appendHistory=False)
            z_raw = (i2[:, :10] - i2[:, 10:]) + b2  # Differential sensing + threshold bias DAC
            z = z_raw * args.gain            # Scale analog currents into standard logit dynamic range
            y_pred = softmax(z)

            # --- BACKWARD PASS (Canonical STE Surrogate Gradients with Threshold Biases) ---
            dz = y_pred - Y_batch  # (bs, 10), natural magnitude in [-1.0, 1.0]
            grad_b2 = np.mean(dz, axis=0)  # (10,)
            grad_W2 = np.dot(h_norm.T, dz) / bs  # (hidden_dim, 10)

            dh_norm = np.dot(dz, W2.T)           # (bs, hidden_dim)
            dh_raw  = (dh_norm / max(1e-6, h_scale)) * (h_raw > 0.0).astype(np.float64)  # Chain rule: d/dh_raw [ReLU(h_raw)/h_scale]
            grad_b1 = np.mean(dh_raw, axis=0)    # (hidden_dim,)
            grad_W1 = np.dot(X_batch.T, dh_raw) / bs  # (64, hidden_dim)

            # Hardware-Aware Adam Optimizer Update with Cosine Annealing & Weight Decay
            opt_step += 1
            grad_W1 = np.clip(grad_W1, -2.0, 2.0)
            grad_W2 = np.clip(grad_W2, -2.0, 2.0)
            grad_b1 = np.clip(grad_b1, -2.0, 2.0)
            grad_b2 = np.clip(grad_b2, -2.0, 2.0)

            m_W1 = beta1 * m_W1 + (1.0 - beta1) * grad_W1
            v_W1 = beta2 * v_W1 + (1.0 - beta2) * (grad_W1 ** 2)
            m_W1_hat = m_W1 / (1.0 - beta1 ** opt_step)
            v_W1_hat = v_W1 / (1.0 - beta2 ** opt_step)
            step_W1 = m_W1_hat / (np.sqrt(v_W1_hat) + eps)
            if args.weight_decay > 0.0:
                step_W1 += args.weight_decay * W1
            W1 = np.clip(W1 - current_lr * step_W1, -1.0, 1.0)

            m_b1 = beta1 * m_b1 + (1.0 - beta1) * grad_b1
            v_b1 = beta2 * v_b1 + (1.0 - beta2) * (grad_b1 ** 2)
            m_b1_hat = m_b1 / (1.0 - beta1 ** opt_step)
            v_b1_hat = v_b1 / (1.0 - beta2 ** opt_step)
            b1 -= current_lr * (m_b1_hat / (np.sqrt(v_b1_hat) + eps))

            m_W2 = beta1 * m_W2 + (1.0 - beta1) * grad_W2
            v_W2 = beta2 * v_W2 + (1.0 - beta2) * (grad_W2 ** 2)
            m_W2_hat = m_W2 / (1.0 - beta1 ** opt_step)
            v_W2_hat = v_W2 / (1.0 - beta2 ** opt_step)
            step_W2 = m_W2_hat / (np.sqrt(v_W2_hat) + eps)
            if args.weight_decay > 0.0:
                step_W2 += args.weight_decay * W2
            W2 = np.clip(W2 - current_lr * step_W2, -1.0, 1.0)

            m_b2 = beta1 * m_b2 + (1.0 - beta1) * grad_b2
            v_b2 = beta2 * v_b2 + (1.0 - beta2) * (grad_b2 ** 2)
            m_b2_hat = m_b2 / (1.0 - beta1 ** opt_step)
            v_b2_hat = v_b2 / (1.0 - beta2 ** opt_step)
            b2 -= current_lr * (m_b2_hat / (np.sqrt(v_b2_hat) + eps))

            # --- PHYSICAL RE-PROGRAMMING ---
            # Periodically program physical memristor filaments across sub-arrays via closed-loop pulse trains
            batches_accumulated += (bs / args.batch_size)
            if batches_accumulated >= args.update_freq or b_end == n_train:
                sim_layer1.updateCrossbarWeights(encode_differential_weights(W1), detailedPrint=False, fineTune=args.fine_tune, silentUpdate=True)
                sim_layer2.updateCrossbarWeights(encode_differential_weights(W2), detailedPrint=False, fineTune=args.fine_tune, silentUpdate=True)
                batches_accumulated = 0.0

            # Metrics
            batch_loss = -np.sum(Y_batch * np.log(y_pred + 1e-12))
            batch_correct = np.sum(np.argmax(y_pred, axis=1) == np.argmax(Y_batch, axis=1))

            epoch_loss += batch_loss
            epoch_correct += batch_correct
            total_seen += bs
            cursor = b_end
            step_idx += 1

            # Dynamic CPU Utilization Controller
            cpu_str = ""
            if HAS_PSUTIL and args.target_cpu > 0:
                cpu_util = psutil.cpu_percent(interval=None)
                cpu_str = f" | CPU: {cpu_util:4.1f}%"
                if cpu_util < (args.target_cpu - 6.0):
                    current_parallel = min(args.max_parallel, current_parallel + 1)
                elif cpu_util > (args.target_cpu + 6.0):
                    current_parallel = max(args.min_parallel, current_parallel - 1)

            # Real-time console progress bar
            progress = (total_seen / n_train) * 100.0
            bar_len = int(progress * 0.2)
            bar_str = ('#' * bar_len).ljust(20)
            curr_acc = (epoch_correct / total_seen) * 100.0
            curr_loss = epoch_loss / total_seen
            elapsed_now = time.perf_counter() - epoch_start_time
            curr_speed = total_seen / max(1e-6, elapsed_now)
            sys.stdout.write(f"\r\033[K  Epoch {epoch:02d}/{args.epochs:02d} [{bar_str}] {progress:5.1f}% | Loss: {curr_loss:.4f} | Acc: {curr_acc:5.1f}% | Par: {current_parallel}x | {curr_speed:.0f} s/s")
            sys.stdout.flush()

        epoch_time = time.perf_counter() - epoch_start_time
        avg_loss = epoch_loss / total_seen
        avg_acc = (epoch_correct / total_seen) * 100.0
        bar_full = '#' * 20
        print(f"\r\033[K  >> Epoch {epoch:02d} Complete [{bar_full}] 100.0% | Time: {epoch_time:.2f}s ({total_seen / epoch_time:.0f} s/s) | LR: {current_lr:.4f} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}%\n")

    # 5. Full Hardware Test Evaluation on 10,000 Test Images
    print("==========================================================================")
    print("         EVALUATING TRUE HARDWARE ACCURACY ON HELD-OUT TEST SET           ")
    print("==========================================================================")
    test_batch_size = 1000
    all_preds = []

    t0_eval = time.perf_counter()
    total_test_batches = int(np.ceil(n_test / test_batch_size))
    for t_idx in range(total_test_batches):
        t_start = t_idx * test_batch_size
        t_end = min(t_start + test_batch_size, n_test)
        X_test_chunk = X_test[t_start:t_end]
        bs_test = t_end - t_start

        # Physical Layer 1 tiled inference
        i1 = sim_layer1.passInput(X_test_chunk, detailedPrint=False, appendHistory=False)
        h_raw = (i1[:, :hidden_dim] - i1[:, hidden_dim:]) + b1
        h_act = np.maximum(0.0, h_raw)
        scale_eval = running_h_scale if running_h_scale is None else float(np.percentile(h_act, 95) + 1e-6)
        h_norm = np.clip(h_act / max(1e-6, scale_eval), 0.0, 1.0)

        # Physical Layer 2 tiled inference
        i2 = sim_layer2.passInput(h_norm, detailedPrint=False, appendHistory=False)
        z_raw = (i2[:, :10] - i2[:, 10:]) + b2
        z = z_raw * args.gain
        preds = np.argmax(z, axis=1)
        all_preds.extend(preds.tolist())

        eval_prog = (t_end / n_test) * 100.0
        bar_eval = ('#' * int(eval_prog * 0.2)).ljust(20)
        sys.stdout.write(f"\r\033[K  [Test Inference] [{bar_eval}] {eval_prog:5.1f}% ({t_end}/{n_test} samples)")
        sys.stdout.flush()

    t1_eval = time.perf_counter()
    eval_time = t1_eval - t0_eval
    all_preds = np.array(all_preds)
    total_correct = int(np.sum(all_preds == test_y))
    overall_test_acc = (total_correct / n_test) * 100.0

    print(f"\n\nTest evaluation completed in {eval_time:.2f}s ({n_test / max(1e-6, eval_time):.1f} samples/s)\n")
    print("==========================================================================")
    print("         PER-DIGIT HARDWARE TEST ACCURACY BREAKDOWN (0-9)                ")
    print("==========================================================================")
    print("  Digit       Accuracy        Correct / Total     Visual Distribution")
    print("  " + "-" * 66)
    for digit in range(10):
        mask_d = (test_y == digit)
        total_d = int(np.sum(mask_d))
        if total_d > 0:
            correct_d = int(np.sum(all_preds[mask_d] == digit))
            acc_d = (correct_d / total_d) * 100.0
            bar_d = ('=' * int(acc_d / 5.0)).ljust(20)
            print(f"    [{digit}]        {acc_d:6.2f}%          {correct_d:5d} / {total_d:5d}        [{bar_d}]")
    print("  " + "-" * 66)
    print(f"  OVERALL TEST ACCURACY: {overall_test_acc:6.2f}% ({total_correct} / {n_test})")
    print("==========================================================================")
    print("--- Hardware Crossbar Simulation Successfully Finished ---")
    print("==========================================================================")


if __name__ == "__main__":
    main()
