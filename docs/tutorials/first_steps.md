# Tutorial: First Steps with Neuro-Symbolic AI

This tutorial guides you through the core concepts of the platform: initializing the pipeline, processing an image, and querying the results.

## Prerequisites

Ensure you have installed the package:
```bash
pip install -e .
```

## Step 1: Initialization

The `NeuralSymbolicPipeline` is the easiest way to start. It wraps the detector, scene graph, and reasoning engine into one object.

```python
from nesy.pipeline import NeuralSymbolicPipeline

# use "mock" backends to run without heavy model weights
pipeline = NeuralSymbolicPipeline(
    detector_backend="mock",
    feature_backend="mock"
)
```

## Step 2: Processing Input

In a real application, you would capture an image from a camera. Here we simulate one.

```python
import numpy as np

# Create a random image (H=480, W=640, C=3)
image = np.zeros((480, 640, 3), dtype=np.uint8)

# The pipeline detects objects in the image and populates the Scene Graph
result = pipeline.process_image(image)

print(f"Time taken: {result['processing_time']:.2f} ms")
print(f"Facts generated: {result['num_facts']}")
```

## Step 3: Inspecting the Scene Graph

The Scene Graph holds the state of the world.

```python
# Access the internal scene graph
sg = pipeline.scene_graph

# List all nodes
for node_id, node in sg.nodes.items():
    print(f"Object: {node_id}, Class: {node.attributes.get('class')}")
```

## Step 4: Reasoning

The reasoning engine allows us to query relationships that weren't explicitly detected but inferred.

For example, if the detector sees a "cup" and a "table", the engine might infer `on(cup, table)` based on their coordinates.

```python
# Query for specific relations
on_relations = pipeline.reasoning.query("on")
for src, dst in on_relations:
    print(f"{src} is ON {dst}")

# Natural language query
answer = pipeline.query("What is on the table?")
print(f"Answer: {answer}")
```

## Step 5: VSA Grounding (Advanced)

Vector Symbolic Architectures allow us to reason about object similarities.

```python
# Find objects similar to 'cup_1' based on visual features
similar_objects = pipeline.find_similar_to("cup_1")
print(f"Objects similar to cup_1: {similar_objects}")
```

## Conclusion

You have successfully:
1.  Initialized the neuro-symbolic pipeline.
2.  Processed visual data.
3.  Queried the symbolic state.
4.  Explored neural similarities.

Check out the [Architecture Guide](../architecture/overview.md) for deeper details.
