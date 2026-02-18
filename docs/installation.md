# Installation & Setup

## Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for neural models)
- **ROS 2 Humble/Iron** (optional, for robotics integration)
- **Scallop** (reasoning engine backend)

## Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/neuro-symbolique.git
    cd neuro-symbolique
    ```

2.  **Create Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    **Optional**: Scallop (Recommended for faster reasoning)
    The system includes a pure Python reasoning engine fallback, so you don't *need* to install Scallop. However, for better performance on large graphs:

    ```bash
    # Scallop is not on PyPI, install from source:
    git clone https://github.com/scallop-lang/scallop.git
    cd scallop
    cargo install scallopy
    ```

    Note: if using ROS 2, ensure you have sourced your ROS environment first:
    ```bash
    source /opt/ros/humble/setup.bash
    ```

4.  **Verify Installation**
    Run the test suite to ensure everything is set up correctly:
    ```bash
    pytest tests/unit
    ```

## Configuration

Configuration is managed via YAML files in `configs/`.

- `configs/default.yaml`: Standard configuration for production.
- `configs/minimal.yaml`: Lightweight configuration for testing/development.

To run with a specific config:
```bash
python -m nesy.main --config configs/minimal.yaml
```

## Troubleshooting

- **ImportError: No module named 'nesy'**: Ensure you are in the root directory and the virtual environment is active. You may need to install the package in editable mode: `pip install -e .`
- **CUDA Errors**: Check your PyTorch installation matches your CUDA version.
