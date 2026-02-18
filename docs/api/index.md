# API Reference

The `nesy` package is organized into several key submodules.

## `nesy.pipeline`

The main entry point for using the platform.

- **`NeuralSymbolicPipeline`**: Orchestrates perception, reasoning, and VSA.
    - `process_image(image, ...)`: Updates the scene graph from visual input.
    - `query(text)`: Answers natural language questions.
    - `infer(term)`: Checks if a logical term is true.

## `nesy.perception`

Handles sensor data processing.

- **`PerceptionPipeline`**: Detection -> Feature Extraction -> Graph Update.
- **`ObjectDetector`**: Wrapper for YOLO/other detectors.
- **`FeatureExtractor`**: Wrapper for CLIP/ResNet.

## `nesy.reasoning`

The cognitive core.

- **`ReasoningEngine`**: Interface to Scallop logic program.
    - `add_fact(relation, *args)`: Adds a fact to the database.
    - `query(relation)`: Returns all tuples for a relation.
- **`VisualReasoningAgent`**: specialized agent for VQA.
- **`NeuralGrounding`**: Projects embeddings to VSA space.

## `nesy.world_model`

Data structures for world state.

- **`SceneGraph`**: Graph of objects and relations.
- **`KnowledgeGraph`**: Semantic ontology layer.
- **`TemporalGraph`**: History of state changes.

## `nesy.agents`

Autonomous agent implementations.

- **`AutonomousAgent`**: The main agent loop (Sense-Think-Act).
- **`Goal`**: Planning objective (e.g., target predicates).

## `nesy.robots`

Robotics integration (hardware/ROS).

- **`RobotNavigator`**: High-level navigation control.
- **`ObjectManipulator`**: Pick-and-place control.
