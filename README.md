# NumPy Matrix Operations

**Veda Technology Internship | AI & ML Track | Level 1, Day 14 (Task 14)**
**Submitted by:** J Sanjay, AI & Data Science

## Objective

Create two matrices using NumPy and perform addition, subtraction, matrix multiplication and transpose operations, to understand the basic matrix operations used in machine learning.

## Tools Used

- Python
- NumPy
- Jupyter Notebook

## Approach

1. Created two 2x2 matrices manually using `np.array()`.
2. Performed addition (`A + B`) and subtraction (`A - B`).
3. Performed matrix multiplication using the `@` operator.
4. Found the transpose of both matrices using `.T`.
5. Compared matrix multiplication with element-wise multiplication (`A * B`).

## Code

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:\n", A)
print("B:\n", B)
print("Addition:\n", A + B)
print("Subtraction:\n", A - B)
print("Matrix Multiplication:\n", A @ B)
print("Transpose of A:\n", A.T)
print("Transpose of B:\n", B.T)
```

## Results

| Operation | Result |
|---|---|
| A + B | [[6, 8], [10, 12]] |
| A - B | [[-4, -4], [-4, -4]] |
| A @ B | [[19, 22], [43, 50]] |
| A * B (element-wise) | [[5, 12], [21, 32]] |
| Transpose of A | [[1, 3], [2, 4]] |
| Transpose of B | [[5, 7], [6, 8]] |

## Matrix Multiplication Explained

Each row of the first matrix is multiplied element by element with each column of the second matrix, and the products are summed to give one entry of the result.

Example: `result[0][0] = (1 x 5) + (2 x 7) = 19`

The number of columns in the first matrix must equal the number of rows in the second matrix. Also, `A @ B` is generally not equal to `B @ A`.

### Element-wise vs Matrix Multiplication

- `A * B` multiplies values at the same position.
- `A @ B` takes the dot product of rows and columns.

## Why Matrices Matter in Machine Learning

- Datasets are stored as matrices (rows are samples, columns are features).
- Linear regression and neural network layers are computed as `X @ W + b`.
- Matrix operations process many samples at once and run efficiently on CPUs and GPUs.

## Outcome

All required operations were completed successfully and the outputs were verified. This task built a clear understanding of the matrix fundamentals behind machine learning algorithms.

## Repository Contents

- `Task14_Matrix_Operations.ipynb`: Jupyter Notebook with code and outputs
- `Task14_NumPy_Matrix_Operations_Report.pdf`: Task report
- `README.md`: Project description
-
