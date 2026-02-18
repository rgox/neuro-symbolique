# Reasoning Engine

The reasoning engine forms the cognitive layer of the platform, combining symbolic logic with vector-based representations.

## 1. Scallop Logic Engine
We use **Scallop**, a neuro-symbolic logic language, for deducing facts. Scallop extends Datalog with differentiable provenance, allowing us to handle uncertainty from the perception layer.

### Rules
Rules are defined in `.scl` files. Example:
```prolog
% Transitivity of 'in' relation
in(X, Z) :- in(X, Y), in(Y, Z).

% Spatial reasoning
on(X, Y) :- near(X, Y), above(X, Y).
```

### Integration
The `ReasoningEngine` synchronizes with the Scene Graph:
1.  **Export**: Scene Graph nodes and edges are exported as Scallop facts (e.g., `node("cup1")`, `edge("cup1", "table1", "on")`).
2.  **Execute**: Scallop program runs to derive new facts.
3.  **Query**: Users query the engine (e.g., `engine.query("on")`) to get results.

## 2. Vector Symbolic Architectures (VSA)
VSA (or Hyperdimensional Computing) provides a way to represent symbols as high-dimensional vectors. This allows for:
- ** Robustness**: The representation is distributed; bit-flips don't destroy information.
- **Operations**:
    - **Binding ($\otimes$)**: Associates concepts (e.g., `Color` $\otimes$ `Red`).
    - **Bundling ($\oplus$)**: Combines concepts (e.g., `Apple` = `Color` $\otimes$ `Red` $\oplus$ `Shape` $\otimes$ `Round`).

### Implementation
- `VSACodebook`: Manages the mapping between atomic symbols and hypervectors.
- `NeuralGrounding`: Projects neural embeddings (from CLIP) into VSA space to allow symbolic operations on neural features.

## 3. Temporal Reasoning
The system tracks the state of the world over time.
- **Snapshot Scans**: Periodic snapshots of the scene graph are stored.
- **Interval Logic**: Queries like "Was the cup on the table between T1 and T2?" are supported using Allen's Interval Algebra.
