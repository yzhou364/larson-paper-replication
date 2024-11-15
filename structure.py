import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete project directory structure."""
    directories = [
        "src",
        "src/core",
        "src/models",
        "src/queuing",
        "src/storage",
        "src/analysis",
        "src/geometry",
        "src/utils",
        "src/visualization",
        "examples",
        "tests/test_core",
        "tests/test_models",
        "tests/test_queuing",
        "tests/test_analysis",
        "scripts",
        "notebooks",
        "docs/api",
        "docs/examples",
        "docs/theory",
        "data/sample_configs",
        "data/test_data",
        "data/results"
    ]
    
    # Create directories and __init__.py files
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        if not directory.startswith(('docs', 'data')):  # Don't create __init__.py in docs and data
            init_path = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w") as f:
                    f.write('"""Hypercube Queuing Model Implementation"""\n')

def create_documentation_files():
    """Create documentation files."""
    docs = {
        "README.md": """# Hypercube Queuing Model Implementation
Based on Larson's 1974 paper "A hypercube queuing model for facility location and redistricting in urban emergency services"

## Overview
This project implements the hypercube queuing model described in Larson's 1974 paper.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.models import ZeroLineModel, InfiniteLineModel

# Create and run models
model = ZeroLineModel(N=9, J=18, lambda_rate=1.0)
results = model.run()
```

## Documentation
See `docs/` directory for detailed documentation.
""",
        "LICENSE": """MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge...""",
        "docs/paper_replication.md": """# Paper Replication Guide
Step-by-step guide for replicating Larson's 1974 paper results..."""
    }
    
    for filepath, content in docs.items():
        with open(filepath, "w") as f:
            f.write(content)

def create_requirements():
    """Create requirements.txt file."""
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "jupyter>=1.0.0",
        "pytest>=6.2.0",
        "typing>=3.7.4",
        "plotly>=5.0.0",  # For interactive visualizations
        "sphinx>=4.0.0"   # For documentation
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))

def create_setup_file():
    """Create setup.py file."""
    setup_content = """from setuptools import setup, find_packages

setup(
    name="hypercube_model",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Implementation of Larson's Hypercube Queuing Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hypercube_model",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
"""
    with open("setup.py", "w") as f:
        f.write(setup_content)

def create_source_files():
    """Create all source files."""
    files = {
        # Core components
        "src/core/binary_sequences.py": binary_sequences_content,
        "src/core/transition_matrix.py": transition_matrix_content,
        "src/core/steady_state.py": steady_state_content,
        "src/core/performance.py": performance_content,
        
        # Models
        "src/models/base_model.py": base_model_content,
        "src/models/zero_line_model.py": zero_line_content,
        "src/models/infinite_line_model.py": infinite_line_content,
        
        # Queuing
        "src/queuing/queue_handler.py": queue_handler_content,
        "src/queuing/dispatch_policies.py": dispatch_policies_content,
        "src/queuing/priority_handler.py": priority_handler_content,
        
        # Storage
        "src/storage/matrix_compression.py": matrix_compression_content,
        "src/storage/state_mapping.py": state_mapping_content,
        
        # Analysis
        "src/analysis/metrics.py": metrics_content,
        "src/analysis/comparison.py": comparison_content,
        "src/analysis/sensitivity.py": sensitivity_content,
        "src/analysis/optimization.py": optimization_content,
        
        # Geometry
        "src/geometry/atoms.py": atoms_content,
        "src/geometry/districts.py": districts_content,
        "src/geometry/travel_time.py": travel_time_content,
        
        # Utils
        "src/utils/validation.py": validation_content,
        "src/utils/config.py": config_content,
        "src/utils/logging_utils.py": logging_content,
        
        # Visualization
        "src/visualization/figures.py": figures_content,
        "src/visualization/plotting_utils.py": plotting_utils_content,
        "src/visualization/interactive.py": interactive_content,
        
        # Examples
        "examples/linear_command.py": linear_command_content,
        "examples/custom_geometries.py": custom_geometries_content,
        "examples/optimization_examples.py": optimization_examples_content,
        
        # Scripts
        "scripts/run_analysis.py": run_analysis_content,
        "scripts/generate_report.py": generate_report_content,
        "scripts/optimization_study.py": optimization_study_content,
        
        # Tests
        "tests/test_core/test_binary_sequences.py": test_binary_sequences_content,
        "tests/test_models/test_zero_line.py": test_zero_line_content,
        "tests/test_queuing/test_dispatch.py": test_dispatch_content,
        
        # Main entry point
        "main.py": main_content
    }
    
    for filepath, content in files.items():
        with open(filepath, "w") as f:
            f.write(content)

def main():
    """Main setup function."""
    print("Setting up Hypercube Model project...")
    
    # Create directory structure and __init__.py files
    create_directory_structure()
    print("Created directory structure and __init__.py files")
    
    # Create documentation files
    create_documentation_files()
    print("Created documentation files")
    
    # Create requirements.txt
    create_requirements()
    print("Created requirements.txt")
    
    # Create setup.py
    create_setup_file()
    print("Created setup.py")
    
    # Create source files
    create_source_files()
    print("Created source files")
    
    print("\nSetup complete! Project structure has been created.")
    print("\nTo get started:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Install package in development mode: pip install -e .")
    print("3. Run the model: python main.py")

if __name__ == "__main__":
    main()