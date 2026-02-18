# Neuro-Symbolic AI Platform

A unified framework for integrating neural perception, symbolic reasoning, and autonomous planning for robotics and AI agents.

## Overview

The `nesy` platform combines the robustness of deep learning with the interpretability of symbolic logic. It enables agents to:

1.  **Perceive**: Detect objects and extract features using neural models (YOLO, CLIP, etc.).
2.  **Represent**: Map perceptions to a structured Knowledge Graph and Vector Symbolic Architecture (VSA).
3.  **Reason**: Infer new knowledge using logical rules (Scallop) and temporal reasoning.
4.  **Plan**: Generate goal-oriented action sequences using PDDL planning.
5.  **Act**: Execute plans via ROS 2 or simulation, with visual servoing and error recovery.

## Documentation Structure

- **[Installation](installation.md)**: Setup instructions, dependencies, and environment configuration.
- **[Quick Start](quickstart.md)**: Run your first query and explore the platform in minutes.
- **Architecture**:
    - [System Overview](architecture/overview.md)
    - [Perception Module](architecture/perception.md)
    - [Reasoning Engine](architecture/reasoning.md)
    - [Planning & Navigation](architecture/planning.md)
- **API Reference**:
    - [API Index](api/index.md)
- **Tutorials**:
    - [First Steps](tutorials/first_steps.md)

## Key Features

- **Hybrid Intelligence**: Seamless integration of neural networks (for pattern recognition) and symbolic logic (for rigorous reasoning).
- **Temporal Reasoning**: Track object states and relationships over time (e.g., "Was the cup on the table before the robot moved?").
- **Vector Symbolic Architectures (VSA)**: Represent complex concepts as high-dimensional vectors, enabling robust symbol grounding.
- **ROS 2 Integration**: Native support for robotic control and sensor feedback.
- **Extensible**: Easily add new models, rules, or planning domains.

## License

MIT License. See `LICENSE` for details.
