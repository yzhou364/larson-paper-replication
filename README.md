# Hypercube Queuing Model Implementation

This repository implements Larson's 1974 paper "A hypercube queuing model for facility location and redistricting in urban emergency services". The implementation includes both zero-line and infinite-line capacity models with comprehensive analysis and visualization tools.

## Project Structure

```
larsonPaperReplication/
├── src/
│   ├── core/
│   │   ├── binary_sequences.py     # Hypercube state sequence generation
│   │   ├── performance.py          # Performance metrics calculation
│   │   ├── steady_state.py         # Steady state probability computation
│   │   └── transition_matrix.py    # Transition rate matrix handling
│   ├── models/
│   │   ├── base_model.py          # Abstract base model class
│   │   ├── zero_line_model.py     # Zero-line capacity implementation
│   │   └── infinite_line_model.py # Infinite-line capacity implementation
│   ├── analysis/
│   │   ├── comparison.py          # Model comparison utilities
│   │   ├── optimization.py        # System optimization tools
│   │   └── sensitivity.py         # Sensitivity analysis
│   ├── utils/
│   │   ├── config.py             # Configuration management
│   │   └── logging_utils.py      # Logging utilities
│   └── visualization/
│       ├── figures.py            # Static visualization (matplotlib)
│       └── interactive.py        # Interactive visualization (plotly)
├── main.py                       # Main entry point
├── setup.py                      # Package setup
└── requirements.txt              # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/larsonPaperReplication.git
cd larsonPaperReplication
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the default analysis with:
```bash
python main.py
```

### Configuration Options

The model supports various configuration parameters that can be set in the `config.py` file or passed through command line arguments:

#### System Parameters
- `N`: Number of units (default: 9)
- `J`: Number of atoms (default: 18)
- `lambda_rate`: Call arrival rate (default: 4.5)
- `mu_rate`: Service rate (default: 1.0)

Example with custom parameters:
```python
config = ModelConfig(
    system=SystemConfig(
        N=12,              # 12 units
        J=24,              # 24 atoms
        lambda_rate=6.0,   # Higher arrival rate
        mu_rate=1.2        # Faster service rate
    ),
    geometry=GeometryConfig(
        district_length=1.0,
        is_grid=False
    ),
    computation=ComputationConfig(
        max_iterations=1000,
        tolerance=1e-10
    ),
    output=OutputConfig(
        save_path="results",
        plot_results=True
    )
)
```

#### Analysis Parameters
- `rho_values`: System utilization levels (default: np.linspace(0.1, 0.9, 9))
- Model types: 'zero_line' or 'infinite_line'
- Dispatch policies: 'mcm', 'district', or 'workload'

### Output

The program generates several outputs in the specified results directory:

1. **Static Visualizations** (`figures/`)
   - Workload distributions
   - Response time comparisons
   - Queue length analysis (infinite-line model)
   - Performance comparisons

2. **Interactive Dashboards** (`dashboard/`)
   - System performance overview
   - Queue analysis dashboard
   - Workload heatmaps

3. **Analysis Report** (`analysis_report.md`)
   - System configuration
   - Performance metrics
   - Model comparisons
   - Recommendations

### Example Configurations

Here are some example configurations for different scenarios:

1. **Small System**
```python
config = ModelConfig(
    system=SystemConfig(
        N=4,              # 4 units
        J=8,              # 8 atoms
        lambda_rate=2.0,
        mu_rate=1.0
    )
)
```

2. **High-Load System**
```python
config = ModelConfig(
    system=SystemConfig(
        N=9,
        J=18,
        lambda_rate=8.0,  # Higher arrival rate
        mu_rate=1.0
    )
)
```

3. **Grid Configuration**
```python
config = ModelConfig(
    system=SystemConfig(
        N=16,            # 16 units
        J=32,            # 32 atoms
        lambda_rate=8.0,
        mu_rate=1.0
    ),
    geometry=GeometryConfig(
        is_grid=True,
        rows=4,          # 4x4 grid
        cols=4
    )
)
```

### Performance Considerations

- **System Size**: Computation time increases exponentially with N (number of units)
- **Convergence**: Higher utilization levels (ρ) may require more iterations
- **Memory Usage**: Large systems may require significant memory for transition matrices

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Larson, R. C. (1974). A hypercube queuing model for facility location and redistricting in urban emergency services. Computers & Operations Research, 1(1), 67-95.

