# ndarray
A NumPy-inspired C library for N-dimensional arrays, supporting machine learning and more..

# Features

## Core Capabilities

- 🔢 **N-dimensional Array Support**
    - Flexible data types (float/double/integer)
    - Memory-efficient storage with smart stride calculation

## Data Operations

- 📥 **Array Creation**
    - CSV file import
    - Random initialization (normal/uniform distribution)
    - Custom shape and type configuration
- 🧮 **Mathematical Functions**
    - Element-wise operations (+, -, *)
    - Matrix multiplication
    - Advanced broadcasting support
    - Comparative operations (>, <, ==)
- 🔄 **Array Transformations**
    - Transposition
    - Subsampling
    - Array concatenation
- 🔧 **Memory Management**
    - Automatic resource cleanup
    - Optimized memory usage

# Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ndarray.git
cd ndarray

# Build library
make
```

This will generate:

- [libndarray.so](http://libndarray.so) - Main library file
- test executable - Demo program

# Usage Guide

## Basic Example

```c
#include "ndarray.h"

int main() {
    // Create 2x3 array
    uint64_t dims[] = {2, 3};
    ndarray_t* arr = ndarray_create(dims, 2, 'd');

    // Generate random values
    ndarray_t* rand_arr = ndarray_random_normal(dims, 2, 0.0, 1.0, 'd');

    // Perform operations
    ndarray_t* result = ndarray_add(arr, rand_arr);

    // Cleanup
    ndarray_free(arr);
    ndarray_free(rand_arr);
    ndarray_free(result);
    return 0;
}
```

## CSV Import

```c
ndarray_t* data = ndarray_from_csv("dataset.csv", 'd');
```

## Matrix Operations

```c
uint64_t dims_a[] = {2, 3};
uint64_t dims_b[] = {3, 4};
ndarray_t* a = ndarray_random_normal(dims_a, 2, 0.0, 1.0, 'd');
ndarray_t* b = ndarray_random_normal(dims_b, 2, 0.0, 1.0, 'd');
ndarray_t* product = ndarray_dot(a, b);
```

# API Reference 📚

## Core Functions

| Function | Parameters | Description |
| --- | --- | --- |
| `ndarray_create` | `dims, nd, dtype` | Create array with specified shape |
| `ndarray_free` | `array` | Release array resources |
| `ndarray_from_csv` | `filename, dtype` | Load array from CSV |

## Random Generation

| Function | Parameters | Description |
| --- | --- | --- |
| `ndarray_random_noise` | `dims, nd, mean, noise_std, dtype` | Uniform noise array |
| `ndarray_random_normal` | `dims, nd, mean, std, dtype` | Normal distribution array |

## Operations

| Function | Parameters | Description |
| --- | --- | --- |
| `ndarray_add` | `a, b` | Element-wise addition |
| `ndarray_subtract` | `a, b` | Element-wise subtraction |
| `ndarray_dot` | `a, b` | Matrix multiplication |
| `ndarray_transpose` | `array` | Transpose array |
| `ndarray_concat` | `a, b, axis` | Concatenate arrays |

# Examples 💡

## Broadcasting

```c
uint64_t dims1[] = {3, 1};
uint64_t dims2[] = {1, 4};
ndarray_t* a = ndarray_create(dims1, 2, 'd');
ndarray_t* b = ndarray_create(dims2, 2, 'd');
ndarray_t* result = ndarray_broadcast_add(a, b);
```

## Comparison

```c
ndarray_t* comparison = ndarray_compare(a, b, '>');
```

# Building 🛠️

Include in your project:

```c
#include "ndarray.h"
```

Compile with:

```bash
gcc your_app.c -L. -lndarray -lm -o your_app
```

# Contributing 🤝

1. Fork the repository
2. Create feature branch (`git checkout -b feature/awesome-feature`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push to branch (`git push origin feature/awesome-feature`)
5. Open Pull Request

# License 📄

MIT License
