# Hypercube Queuing Model Implementation
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
