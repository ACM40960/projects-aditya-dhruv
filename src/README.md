# Source Modules

| File | Description |
|---|---|
| features.py | Feature engineering: rolling features, availability flag, per-90 metrics, position encoding |
| optimiser.py | ILP squad optimiser using PuLP with budget, positional and club constraints |

## Usage

```python
from src.features import add_rolling_features, add_availability_flag
from src.optimiser import optimise_squad, squad_summary
```

## Author
Dhruv Mehta - ACM 40960, University College Dublin
