# Vamana Python Bindings

Python bindings for the Vamana graph-based index using pybind11.

## Requirements

- C++17 compiler
- CMake >= 3.10
- Python 3.8 only(maybe need to update cmakelists to adjust)
- pybind11

## Installation

```bash
make pybind
```

## Usage

use the vamana data-gen script `data_processor_vamana` and the corresponding configs,
such as `config/partitionConfigVamana.yaml`
