# Tensors: shape, dtype, and broadcasting

The first experiment creates a `2 x 3` floating-point matrix and multiplies it
by a length-three scale vector. PyTorch aligns the vector with the matrix's
last dimension, so the same three scale values apply independently to both
rows. Broadcasting changes neither input in place.

```mermaid
flowchart LR
    A["inputs<br/>shape: 2 x 3<br/>dtype: float32"]
    B["scale<br/>shape: 3<br/>dtype: float32"]
    C["element-wise multiply<br/>broadcast scale over rows"]
    D["scaled<br/>shape: 2 x 3<br/>dtype: float32"]
    A --> C
    B --> C
    C --> D
```

Run the executable example:

```bash
python -m tiny_transformer.experiments.tensors
```

The fixed local random generator makes the output reproducible without
changing PyTorch's process-wide random seed.
