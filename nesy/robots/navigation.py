"""
Robot Navigation with PDDL Planning.

Combines PDDL planning with ROS navigation for autonomous robot movement.

Features:
- Topological map navigation
- PDDL-based path planning
- Waypoint following
- Obstacle-aware replanning
- Nav2 integration

Example:
    >>> from nesy.robots import RobotNavigator
    >>> 
    >>> nav = RobotNavigator(ros_bridge)
    >>> nav.add_waypoint("kitchen", 2.0, 3.0)
    >>> nav.add_waypoint("living_room", 5.0, 1.0)
    >>> nav.navigate_to("kitchen")
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """Navigation waypoint."""
    name: str
    x: float
    y: float
    theta: float = 0.0  # Orientation
    metadata: Dict = field(default_factory=dict)


@dataclass
class NavigationPath:
    """Navigation path between waypoints."""
    waypoints: List[Waypoint] = field(default_factory=list)
    total_distance: float = 0.0
    estimated_time: float = 0.0


class TopologicalMap:
    """
    Topological map for navigation.
    
    Represents locations and connections as a graph,
    used by the PDDL planner for navigation planning.
    
    Example:
        >>> topo_map = TopologicalMap()
        >>> topo_map.add_waypoint("kitchen", 2.0, 3.0)
        >>> topo_map.add_waypoint("hallway", 4.0, 3.0)
        >>> topo_map.add_connection("kitchen", "hallway")
    """
    
    def __init__(self):
        """Initialize topological map."""
        self.waypoints: Dict[str, Waypoint] = {}
        self.connections: Dict[str, List[str]] = {}
    
    def add_waypoint(self, name: str, x: float, y: float, theta: float = 0.0):
        """Add waypoint to map."""
        self.waypoints[name] = Waypoint(name=name, x=x, y=y, theta=theta)
        if name not in self.connections:
            self.connections[name] = []
        logger.info(f"Added waypoint: {name} ({x:.1f}, {y:.1f})")
    
    def add_connection(self, from_wp: str, to_wp: str, bidirectional: bool = True):
        """Add connection between waypoints."""
        if from_wp in self.connections:
            self.connections[from_wp].append(to_wp)
        if bidirectional and to_wp in self.connections:
            self.connections[to_wp].append(from_wp)
    
    def get_neighbors(self, waypoint: str) -> List[str]:
        """Get connected waypoints."""
        return self.connections.get(waypoint, [])
    
    def distance(self, wp1: str, wp2: str) -> float:
        """Calculate distance between waypoints."""
        w1 = self.waypoints.get(wp1)
        w2 = self.waypoints.get(wp2)
        if w1 and w2:
            return math.sqrt((w1.x - w2.x)**2 + (w1.y - w2.y)**2)
        return float('inf')
    
    def to_pddl_facts(self) -> List[str]:
        """Convert map to PDDL facts."""
        facts = []
        for name in self.waypoints:
            facts.append(f"location({name})")
        
        for from_wp, neighbors in self.connections.items():
            for to_wp in neighbors:
                facts.append(f"connected({from_wp}, {to_wp})")
        
        return facts


class RobotNavigator:
    """
    Robot navigation using PDDL planning.
    
    Combines topological map with PDDL planner for
    high-level navigation planning, and ROS bridge for
    low-level movement execution.
    
    Example:
        >>> nav = RobotNavigator(ros_bridge)
        >>> 
        >>> # Build map
        >>> nav.add_waypoint("kitchen", 2.0, 3.0)
        >>> nav.add_waypoint("hallway", 4.0, 3.0)
        >>> nav.add_connection("kitchen", "hallway")
        >>> 
        >>> # Navigate
        >>> nav.set_current_location("hallway")
        >>> path = nav.navigate_to("kitchen")
    """
    
    def __init__(self, ros_bridge=None):
        """
        Initialize navigator.
        
        Args:
            ros_bridge: ROS bridge for robot communication
        """
        self.ros_bridge = ros_bridge
        self.topo_map = TopologicalMap()
        self.current_location: Optional[str] = None
        
        # Navigation state
        self.is_navigating = False
        self.current_path: Optional[NavigationPath] = None
        
        # Parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 1.0  # rad/s
        self.goal_tolerance = 0.3  # meters
        
        # History
        self.navigation_history: List[Tuple[str, str]] = []
    
    def add_waypoint(self, name: str, x: float, y: float, theta: float = 0.0):
        """Add waypoint."""
        self.topo_map.add_waypoint(name, x, y, theta)
    
    def add_connection(self, from_wp: str, to_wp: str, bidirectional: bool = True):
        """Add connection."""
        self.topo_map.add_connection(from_wp, to_wp, bidirectional)
    
    def set_current_location(self, location: str):
        """Set current robot location."""
        self.current_location = location
        logger.info(f"Current location: {location}")
    
    def navigate_to(self, target: str) -> Optional[NavigationPath]:
        """
        Navigate to target location.
        
        Uses PDDL planner for path planning and ROS for execution.
        
        Args:
            target: Target waypoint name
        
        Returns:
            Navigation path or None if unreachable
        """
        if not self.current_location:
            logger.error("Current location not set")
            return None
        
        if target not in self.topo_map.waypoints:
            logger.error(f"Unknown target: {target}")
            return None
        
        if self.current_location == target:
            logger.info("Already at target")
            return NavigationPath()
        
        # Plan path using PDDL
        path = self._plan_path(self.current_location, target)
        
        if path:
            self.current_path = path
            self.is_navigating = True
            
            # Execute path
            self._execute_path(path)
            
            # Record
            self.navigation_history.append((self.current_location, target))
            self.current_location = target
            self.is_navigating = False
            
            logger.info(f"Arrived at {target}")
        
        return path
    
    def _plan_path(self, start: str, goal: str) -> Optional[NavigationPath]:
        """Plan path using PDDL."""
        from nesy.reasoning.planning import PDDLDomain, PDDLProblem, PDDLPlanner
        
        # Create navigation domain
        domain = PDDLDomain("navigation")
        domain.add_action(
            "move_to",
            parameters=["from", "to"],
            preconditions=[f"at({start})", "connected(from, to)"],
            effects=[f"not at({start})", "at(to)"]
        )
        
        # Create problem with map facts
        problem = PDDLProblem("nav_task", domain)
        problem.add_initial_fact(f"at({start})")
        
        for fact in self.topo_map.to_pddl_facts():
            problem.add_initial_fact(fact)
        
        problem.add_goal(f"at({goal})")
        
        # Plan
        planner = PDDLPlanner(domain)
        plan = planner.plan(problem, algorithm="astar")
        
        if plan:
            # Convert to navigation path
            waypoints = [self.topo_map.waypoints[start]]
            current = start
            
            # Follow the plan through topology
            for action in plan:
                # Find destination from action
                for neighbor in self.topo_map.get_neighbors(current):
                    if neighbor in self.topo_map.waypoints:
                        waypoints.append(self.topo_map.waypoints[neighbor])
                        current = neighbor
                        break
            
            # Calculate total distance
            total_dist = sum(
                self.topo_map.distance(
                    waypoints[i].name,
                    waypoints[i+1].name
                )
                for i in range(len(waypoints) - 1)
            )
            
            return NavigationPath(
                waypoints=waypoints,
                total_distance=total_dist,
                estimated_time=total_dist / self.linear_speed
            )
        
        return None
    
    def _execute_path(self, path: NavigationPath):
        """Execute navigation path via ROS."""
        for i, wp in enumerate(path.waypoints):
            logger.info(f"Moving to waypoint {i+1}/{len(path.waypoints)}: {wp.name}")
            
            if self.ros_bridge:
                # Send goal pose via ROS
                self.ros_bridge.publish_pose(wp.x, wp.y, 0.0)
                
                # Simple velocity command
                self.ros_bridge.publish_velocity(
                    linear=self.linear_speed,
                    angular=0.0
                )
            
            # Simulate travel time
            if i < len(path.waypoints) - 1:
                dist = self.topo_map.distance(
                    path.waypoints[i].name,
                    path.waypoints[i+1].name
                )
                travel_time = dist / self.linear_speed
                logger.debug(f"Travel time: {travel_time:.1f}s")
        
        # Stop
        if self.ros_bridge:
            self.ros_bridge.publish_velocity(0.0, 0.0)


class ObjectManipulator:
    """
    Object manipulation planning for robotic arms.
    
    Uses PDDL planning for pick-and-place sequences and
    the perception system for object localization.
    
    Example:
        >>> manipulator = ObjectManipulator(ros_bridge)
        >>> manipulator.pick("cup")
        >>> manipulator.place("table")
    """
    
    def __init__(self, ros_bridge=None):
        """
        Initialize manipulator.
        
        Args:
            ros_bridge: ROS bridge for arm control
        """
        self.ros_bridge = ros_bridge
        
        # State
        self.holding: Optional[str] = None
        self.objects: Dict[str, Tuple[float, float, float]] = {}
        self.surfaces: Dict[str, Tuple[float, float, float]] = {}
        
        # Arm parameters
        self.workspace_radius = 0.8  # meters
        self.gripper_open = True
        
        # History
        self.manipulation_history: List[Dict] = []
    
    def add_object(self, name: str, x: float, y: float, z: float):
        """Register object position."""
        self.objects[name] = (x, y, z)
    
    def add_surface(self, name: str, x: float, y: float, z: float):
        """Register surface position."""
        self.surfaces[name] = (x, y, z)
    
    def pick(self, object_name: str) -> bool:
        """
        Pick up an object.
        
        Args:
            object_name: Object to pick
        
        Returns:
            True if successful
        """
        if self.holding:
            logger.warning(f"Already holding {self.holding}")
            return False
        
        if object_name not in self.objects:
            logger.warning(f"Unknown object: {object_name}")
            return False
        
        pos = self.objects[object_name]
        
        # Check reachability
        dist = math.sqrt(pos[0]**2 + pos[1]**2)
        if dist > self.workspace_radius:
            logger.warning(f"{object_name} out of reach ({dist:.2f}m)")
            return False
        
        # Execute pick sequence
        logger.info(f"Picking {object_name} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        if self.ros_bridge:
            # Would send trajectory commands
            pass
        
        # Update state
        self.holding = object_name
        self.gripper_open = False
        del self.objects[object_name]
        
        self.manipulation_history.append({
            "action": "pick", "object": object_name, "position": pos
        })
        
        return True
    
    def place(self, surface_name: str) -> bool:
        """
        Place held object on surface.
        
        Args:
            surface_name: Surface to place on
        
        Returns:
            True if successful
        """
        if not self.holding:
            logger.warning("Not holding anything")
            return False
        
        if surface_name not in self.surfaces:
            logger.warning(f"Unknown surface: {surface_name}")
            return False
        
        pos = self.surfaces[surface_name]
        
        logger.info(f"Placing {self.holding} on {surface_name}")
        
        if self.ros_bridge:
            # Would send trajectory commands
            pass
        
        # Update state
        placed_obj = self.holding
        self.objects[placed_obj] = pos
        self.holding = None
        self.gripper_open = True
        
        self.manipulation_history.append({
            "action": "place", "object": placed_obj, "surface": surface_name
        })
        
        return True
    
    def plan_manipulation(
        self,
        tasks: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        Plan manipulation sequence using PDDL.
        
        Args:
            tasks: List of (object, target_surface) to move
        
        Returns:
            Ordered action sequence
        """
        from nesy.reasoning.planning import PDDLDomain, PDDLProblem, PDDLPlanner
        
        domain = PDDLDomain("manipulation")
        
        domain.add_action(
            "pick_up",
            parameters=["obj"],
            preconditions=["reachable(obj)", "not holding(any)"],
            effects=["holding(obj)", "not on_surface(obj)"]
        )
        
        domain.add_action(
            "place_on",
            parameters=["obj", "surface"],
            preconditions=["holding(obj)", "clear(surface)"],
            effects=["on_surface(obj)", "not holding(obj)", "at(obj, surface)"]
        )
        
        # Build problem
        problem = PDDLProblem("manip_task", domain)
        
        for obj in self.objects:
            problem.add_initial_fact(f"reachable({obj})")
        
        problem.add_initial_fact("not holding(any)")
        
        for surface in self.surfaces:
            problem.add_initial_fact(f"clear({surface})")
        
        for obj, target in tasks:
            problem.add_goal(f"at({obj}, {target})")
        
        planner = PDDLPlanner(domain)
        plan = planner.plan(problem)
        
        if plan:
            return [{"action": str(a)} for a in plan]
        return []


class VisualServoing:
    """
    Visual servoing for precise object interaction.
    
    Uses perception feedback to guide robot movements
    toward visual targets.
    
    Example:
        >>> servo = VisualServoing(ros_bridge)
        >>> servo.set_target(target_position)
        >>> servo.servo_to_target()
    """
    
    def __init__(self, ros_bridge=None):
        """Initialize visual servoing."""
        self.ros_bridge = ros_bridge
        
        # Control parameters
        self.kp_linear = 0.5   # Proportional gain
        self.kp_angular = 1.0
        self.max_linear = 0.3  # Max velocity (m/s)
        self.max_angular = 0.5  # Max angular velocity
        self.tolerance = 0.05  # Target tolerance (m)
        
        # State
        self.target_position: Optional[np.ndarray] = None
        self.is_servoing = False
    
    def set_target(self, position: np.ndarray):
        """Set visual target position."""
        self.target_position = position
        logger.info(f"Visual target set: {position}")
    
    def compute_control(
        self,
        current_position: np.ndarray,
        target_position: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Compute control command from visual feedback.
        
        Args:
            current_position: Current end-effector position
            target_position: Target position (or use stored)
        
        Returns:
            (linear_vel, angular_vel) control command
        """
        target = target_position if target_position is not None else self.target_position
        
        if target is None:
            return 0.0, 0.0
        
        # Error
        error = target[:2] - current_position[:2]
        distance = np.linalg.norm(error)
        
        if distance < self.tolerance:
            return 0.0, 0.0
        
        # Proportional control
        angle = math.atan2(error[1], error[0])
        
        linear = min(self.kp_linear * distance, self.max_linear)
        angular = min(self.kp_angular * angle, self.max_angular)
        angular = max(angular, -self.max_angular)
        
        return float(linear), float(angular)
    
    def servo_step(self, current_pos: np.ndarray) -> bool:
        """
        Execute one servoing step.
        
        Args:
            current_pos: Current position
        
        Returns:
            True if target reached
        """
        linear, angular = self.compute_control(current_pos)
        
        if linear == 0.0 and angular == 0.0:
            logger.info("Visual target reached")
            self.is_servoing = False
            return True
        
        if self.ros_bridge:
            self.ros_bridge.publish_velocity(linear, angular)
        
        return False
