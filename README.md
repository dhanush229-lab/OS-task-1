# Thread-Based Matrix Multiplication Using TensorFlow

This project implements square matrix multiplication with explicit Python worker threads and TensorFlow. The default case is **100 × 100**, which requires **1,000,000 scalar multiplications**.

## Architecture

- **TensorFlow:** performs every scalar multiplication through `tf.multiply()`.
- **Python `ThreadPoolExecutor`:** provides explicit reusable worker threads.
- **Matplotlib:** creates an animation from real sampled execution events.
- **`tf.matmul()`** is used only as an independent correctness reference, not as the primary implementation.

Each output cell `C[i, j]` is assigned to one worker. That worker executes the complete inner loop for `k = 0 ... n-1`; every `A[i,k] × B[k,j]` operation is therefore executed inside the worker-thread context.

We deliberately do **not** create one OS thread per scalar multiplication. One million permanent threads would be impractical. A bounded pool reuses a small number of worker threads for the one-million scalar multiplication tasks.

## Why NumPy appears in the implementation

The input tensors are converted to NumPy once before the worker pool starts. This avoids expensive TensorFlow tensor indexing for every scalar access. The multiplication itself remains TensorFlow `tf.multiply()` inside the worker threads. NumPy is not used to replace the required scalar TensorFlow multiplication.

## Validation

The custom result is compared with `tf.matmul()` using `numpy.allclose` with `rtol=1e-5` and `atol=1e-2`. A small non-zero floating-point difference is expected because the custom implementation accumulates products in a different order from TensorFlow's optimized matrix multiplication.

## Run

Activate the virtual environment, then:

```powershell
python main.py
```

For a quick test:

```powershell
python main.py --size 5 --workers 4 --no-animation
```

For the required 100 × 100 computation without the GUI animation:

```powershell
python main.py --size 100 --workers 4 --no-animation
```

To also run the optional sequential benchmark:

```powershell
python main.py --size 100 --workers 4 --no-animation --benchmark
```

The full animation is written to:

```text
animations/matrix_multiplication.gif
```

## Important performance note

This project intentionally performs one million TensorFlow scalar multiplication calls. That is useful for demonstrating the assignment's thread requirement, but it is not expected to beat optimized `tf.matmul()`. The reference `tf.matmul()` is designed for efficient matrix multiplication and can use TensorFlow's own internal parallel execution.
