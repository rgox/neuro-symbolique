# System Architecture Overview

The `nesy` platform follows a modular, pipeline-based architecture designed to bridge the gap between continuous neural capabilities and discrete symbolic reasoning.

## High-Level Diagram

```mermaid
graph TD
    subgraph Perception
        Sensors[Sensors (Camera/Lidar)] --> Neural[Neural Models (YOLO/CLIP)]
        Neural --> Features[Feature Extraction]
        Features --> SG[Scene Graph Generation]
    end

    subgraph Representation
        SG --> KG[Knowledge Graph]
        Neural --> VSA[VSA Grounding Layer]
        VSA --> KG
        KG --> Temporal[Temporal Graph]
    end

    subgraph Reasoning
        KG --> Engine[Reasoning Engine (Scallop)]
        Engine --> Inference[Inferred Facts]
        Inference --> Query[Query Interface]
    end

    subgraph Action
        Inference --> Planner[PDDL Planner]
        Planner --> Plan[Action Plan]
        Plan --> Executor[Action Executor]
        Executor --> ROS[ROS 2 Controllers]
    end
```

## Core Modules

### 1. Perception (`nesy.perception`)
Responsible for processing raw sensor data into structured representations.
- **Object Detection**: Identifies bounding boxes and class labels.
- **Feature Extraction**: Generates embeddings for detected objects.
- **Scene Graph Construction**: Building the initial graph of objects and spatial relations.

### 2. World Model (`nesy.world_model`)
The central data structure representing the agent's understanding of the world.
- **Scene Graph**: 3D spatial representation of objects and relations.
- **Knowledge Graph**: Semantic layer enriching the scene graph with ontology.
- **Temporal Graph**: Tracks state changes over time.

### 3. Reasoning (`nesy.reasoning`)
The cognitive core of the system.
- **Logic Engine**: Uses Scallop (Datalog) to infer new facts from the world model.
- **VSA (Vector Symbolic Architectures)**: Grounds symbols in high-dimensional vector space for robust similarity and analogy.
- **NL Query**: Translates natural language questions into formal logic queries.

### 4. Planning & Action (`nesy.robots`)
Translates goals into physical actions.
- **PDDL Planner**: Generates symbolic plans (e.g., `pick(cup)`, `move_to(kitchen)`).
- **Navigation**: Path planning and obstacle avoidance via ROS 2.
- **Manipulation**: Control of robotic arms for pick-and-place tasks.

## Data Flow

1.  **Input**: The system receives an image or sensor stream.
2.  **Processing**: Perception models extract object instances and features.
3.  **Grounding**: These instances are "grounded" into the scene graph and VSA codebook.
4.  **Sync**: The reasoning engine synchronizes with the current state of the scene graph.
5.  **Query/Goal**: A user or agent sets a goal or asks a question.
6.  **Inference/Planning**: The engine infers necessary facts or the planner generates a sequence of actions.
7.  **Output**: An answer is returned or an action is executed.
