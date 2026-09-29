import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, CrossbarDense, EpochTracker, suppress_c_stderr

import numpy as np

def main():
    with suppress_c_stderr():
        import tensorflow as tf

    print("==========================================================")
    print("--- TensorFlow Hardware-in-Loop Crossbar Training Loop ---")
    print("==========================================================")
    
    # 1. Small Synthetic Logic Dataset (Learning Linearly Separable Function)
    # Target: Class 1 if Feature0 > Feature1 else Class 0
    X_train = np.array([
        [0.8, 0.2],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.1, 0.9]
    ], dtype=np.float32)
    
    Y_train = np.array([
        [1],
        [0],
        [1],
        [0]
    ], dtype=np.float32)

    # 2. Instantiate the Core Hardware Simulator Array
    sim = MolmemSimulator(instanceName="TF_Hardware_Core")
    sim.addCrossbarMatrix(device_type="default", rows=2, cols=1, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column")
    
    # 3. Build Keras Model
    model = tf.keras.Sequential([
        CrossbarDense(units=1, sim_instance=sim, output_scale=1.0),
        tf.keras.layers.Activation('sigmoid')
    ])
    
    # 4. Compile and Train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("\n>>> STARTING HARDWARE AWARE TRAINING (20 Epochs) <<<")
    with suppress_c_stderr():
        model.fit(X_train, Y_train, epochs=20, batch_size=4, callbacks=[EpochTracker()], verbose=0)
    
    print("\n>>> TRAINING FINISHED <<<")
    print("Final Model Hardware Weights:")
    final_w = model.layers[0].get_weights()[0]
    print(np.round(final_w, 3))
    
    print("\n==========================================================")
    print("--- Physical TensorFlow Training Successfully Completed ---")
    print("==========================================================")

if __name__ == "__main__":
    main()
