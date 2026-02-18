# NeuralSymbolicPipeline

The primary interface for the `nesy` platform.

## Class: `nesy.pipeline.NeuralSymbolicPipeline`

### Initialization

```python
def __init__(
    self,
    detector_backend: str = "mock",
    feature_backend: str = "mock",
    vsa_dim: int = 10000,
    vsa_similarity_threshold: float = 0.5,
    use_uma: bool = True,
    logger: Optional[logging.Logger] = None
)
```

**Parameters:**
- `detector_backend` (str): Backend for object detection. Options: `"mock"`, `"yolo"`, `"faster_rcnn"`.
- `feature_backend` (str): Backend for feature extraction. Options: `"mock"`, `"clip"`, `"resnet"`.
- `vsa_dim` (int): Dimension of VSA hypervectors. Default: `10000`.
- `vsa_similarity_threshold` (float): Similarity threshold for VSA relations. Default: `0.5`.
- `use_uma` (bool): Whether to use Unified Memory Architecture. Default: `True`.

### Methods

#### `process_image`

Process an image through the full pipeline.

```python
def process_image(
    self,
    image: np.ndarray,
    camera_pose: Optional[np.ndarray] = None,
    ground_vsa: bool = True,
    sync_reasoning: bool = True
) -> Dict[str, Any]
```

**Returns:**
Dictionary with keys:
- `detections`: List of detected objects.
- `num_nodes`: Count of scene graph nodes.
- `num_facts`: Count of inferred facts.
- `processing_time`: Execution time in milliseconds.

#### `query`

Answer a natural language query.

```python
def query(self, query_str: str) -> List[str]
```

**Supported Patterns:**
- "What's in the [location]?"
- "Find all [class]"
- "What's on the [object]?"

#### `infer`

Check if a logical term is true.

```python
def infer(self, goal: str) -> bool
```

**Example:**
```python
is_true = pipeline.infer("on('cup1', 'table1')")
```
