# Planning & Navigation

This module enables valid action generation based on the symbolic understanding of the world.

## 1. PDDL Planning
We use **PDDL (Planning Domain Definition Language)** to define the physics of the world and valid actions.

### Domain
Automatically generated or manually defined in `pddl/domain.pddl`:
```lisp
(define (domain robot-domain)
  (:action pick
    :parameters (?obj ?loc)
    :precondition (and (at ?obj ?loc) (at-robot ?loc) (empty-hand))
    :effect (and (holding ?obj) (not (at ?obj ?loc)) (not (empty-hand)))
  )
)
```

### Problem Generation
The `AutonomousAgent` dynamically generates a `.pddl` problem file from the current state of the logic engine (e.g., converting all `on(a, b)` facts to PDDL init state).

### Planners
We support:
- **A* Search**: Internal Python implementation for simple problems.
- **External Solvers**: Fast Downward (optional integration).

## 2. Navigation
Navigation handles moving the robot base to specific coordinates.

### Topological Map
The environment is represented as a graph of **Waypoints** (e.g., "kitchen", "living_room").
- **Nodes**: (x, y, theta) coordinates.
- **Edges**: Navigable paths.

### ROS 2 Integration
The `RobotNavigator` sends goals to the ROS 2 Navigation stack (Nav2).
- **Global Planner**: Plans path across the map.
- **Local Planner**: Avoids dynamic obstacles.

## 3. Manipulation
The `ObjectManipulator` handles arm trajectories.
- **Pick**: Approach -> Grasp -> Lift.
- **Place**: Approach -> Lower -> Release.
- **Visual Servoing**: Uses camera feedback to fine-tune the end-effector position before grasping.
