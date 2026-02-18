# Perception Module

The perception module is the gateway between the physical world and the symbolic system. It transforms raw sensor data into a structured **Scene Graph**.

## Components

### 1. Object Detection
We use **YOLOv8** (or mock backends for testing) to detect objects in 2D images.
- **Input**: RGB Image $(H, W, 3)$
- **Output**: List of bounding boxes $(x_1, y_1, x_2, y_2)$, class labels, and confidence scores.

### 2. Feature Extraction
For every detected object, we extract a high-dimensional feature vector using **CLIP** (Contrastive Language-Image Pre-Training).
- **Purpose**: To enable semantic search and "zero-shot" classification.
- **Process**: Crop the object from the image $\to$ Pass through CLIP visual encoder $\to$ Obtain 512-dim embedding.

### 3. Scene Graph Generation
The detections are assembled into a Scene Graph.
- **Nodes**: Represent objects (e.g., `cup_1`, `table_1`). Attributes include class, position, color, and neural embedding.
- **Edges**: Represent spatial relationships (e.g., `on`, `in`, `near`). These are computed heuristically (e.g., if bbox A is inside bbox B, relationship is `in`) or via depth estimation.

## Configuration

In `configs/default.yaml`:
```yaml
perception:
  detector: "yolo"          # or "mock"
  feature_extractor: "clip" # or "resnet", "mock"
  resolution: [640, 480]
  confidence_threshold: 0.5
```

## Usage

```python
from nesy.perception import PerceptionPipeline

perception = PerceptionPipeline()
result = perception.process_image(image)

print(result.detections)
# [Detection(label='cup', bbox=[...], confidence=0.95)]
```
