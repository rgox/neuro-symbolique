# Quick Start Guide

This guide will help you run your first neuro-symbolic pipeline and perform a simple query.

## 1. Minimal Example

Create a file named `demo.py`:

```python
import numpy as np
from nesy.pipeline import NeuralSymbolicPipeline

# Initialize the pipeline (mock backends for speed)
pipeline = NeuralSymbolicPipeline(
    detector_backend="mock",
    feature_backend="mock"
)

# create a dummy image (usually this would come from a camera)
image = np.zeros((480, 640, 3), dtype=np.uint8)

# Process the image
# This runs object detection, updates the scene graph, and syncs reasoning
result = pipeline.process_image(image)

print(f"Detected {len(result['detections'])} objects")

# Run a natural language query
# (Mock detector puts random objects, or use real image for real results)
# For this example, let's inject a known fact manually to test querying
pipeline.scene_graph.add_object("cup_1", label="cup", confidence=0.9)
pipeline.reasoning.sync_from_scene_graph()

# Query: "Find all cups"
cups = pipeline.query("Find all cups")
print(f"Found cups: {cups}")

# Query: "What's in the scene?" (custom logic)
facts = pipeline.reasoning.query("object")
print(f"All objects: {facts}")
```

Run it:
```bash
python demo.py
```

## 2. Running the Autonomous Agent

For a more complex example involving planning, you can use the autonomous agent:

```python
from nesy.agents import AutonomousAgent, Goal

agent = AutonomousAgent()

# specific goal
goal = Goal("Move cup to table", target_predicates=["on(cup, table)"])

# Run the agent loop (requires valid perception inputs in real usage)
# agent.run(goal)
```

## 3. Using the CLI

The platform provides a CLI for common tasks.

**Run System Check:**
```bash
python -m nesy.tools.main --check
```

**Visualize Scene Graph:**
```bash
python -m nesy.tools.viz --scene-graph
```
