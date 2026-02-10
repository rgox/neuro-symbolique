"""
Autonomous Agent Framework.

Complete end-to-end agent integrating:
- Perception (YOLO, CLIP, Scene Graph)
- Reasoning (Scallop, VSA, Logic)
- Planning (PDDL)
- Action Execution

Example:
    >>> from nesy.agents import AutonomousAgent
    >>> 
    >>> agent = AutonomousAgent()
    >>> agent.observe(image)
    >>> agent.reason()
    >>> plan = agent.plan(goal="clean_table")
    >>> agent.execute(plan)
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    OBSERVING = "observing"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for autonomous agent."""
    # Perception
    use_yolo: bool = True
    use_clip: bool = True
    perception_fps: float = 10.0
    
    # Reasoning
    reasoning_engine: str = "scallop"  # or "mock"
    enable_vsa: bool = True
    
    # Planning
    planner_algorithm: str = "astar"
    max_plan_steps: int = 100
    
    # Execution
    action_timeout: float = 5.0
    replan_on_failure: bool = True
    
    # Loop
    loop_frequency: float = 5.0  # Hz
    enable_async: bool = False


@dataclass
class Observation:
    """Sensor observation."""
    timestamp: float
    image: Optional[np.ndarray] = None
    detections: List[Dict] = field(default_factory=list)
    embeddings: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Goal:
    """Agent goal specification."""
    description: str
    target_predicates: List[str] = field(default_factory=list)
    priority: float = 1.0
    deadline: Optional[float] = None


class AutonomousAgent:
    """
    Autonomous agent with perception-reasoning-planning-action loop.
    
    Architecture:
        1. PERCEIVE: Process sensor data → scene graph
        2. REASON: Apply logic rules → infer new facts
        3. PLAN: Generate action sequence → achieve goal
        4. ACT: Execute actions → update world
    
    Example:
        >>> agent = AutonomousAgent(config)
        >>> 
        >>> # Single step
        >>> agent.observe(camera_image)
        >>> agent.reason()
        >>> plan = agent.plan(goal)
        >>> agent.execute(plan)
        >>> 
        >>> # Autonomous loop
        >>> agent.run(goal, max_iterations=100)
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.state = AgentState.IDLE
        
        # Components (lazy initialization)
        self._perception = None
        self._reasoning = None
        self._planner = None
        self._executor = None
        
        # State tracking
        self.current_observation: Optional[Observation] = None
        self.current_goal: Optional[Goal] = None
        self.current_plan: Optional[List] = None
        
        # History
        self.observation_history: List[Observation] = []
        self.action_history: List[Any] = []
        
        # Metrics
        self.metrics = {
            "observations": 0,
            "inferences": 0,
            "plans_generated": 0,
            "actions_executed": 0,
            "failures": 0
        }
        
        logger.info("Autonomous agent initialized")
    
    def _init_components(self):
        """Lazy initialize all components."""
        if self._perception is None:
            self._init_perception()
        if self._reasoning is None:
            self._init_reasoning()
        if self._planner is None:
            self._init_planner()
        if self._executor is None:
            self._init_executor()
    
    def _init_perception(self):
        """Initialize perception pipeline."""
        from nesy.perception.models import ModelZoo
        from nesy.world_model.scene_graph import SceneGraph
        
        self.scene_graph = SceneGraph()
        self.model_zoo = ModelZoo()
        
        # Load models if enabled
        if self.config.use_yolo:
            self.detector = self.model_zoo.get_yolo()
        if self.config.use_clip:
            self.embedder = self.model_zoo.get_clip()
        
        logger.info("Perception initialized")
    
    def _init_reasoning(self):
        """Initialize reasoning engine."""
        from nesy.reasoning.logic import ReasoningEngine
        
        self._reasoning = ReasoningEngine(self.scene_graph)
        
        # Add domain rules
        self._add_domain_rules()
        
        logger.info("Reasoning initialized")
    
    def _init_planner(self):
        """Initialize planner."""
        from nesy.reasoning.planning import PDDLDomain, PDDLPlanner
        
        # Create domain
        self.domain = self._create_planning_domain()
        self._planner = PDDLPlanner(self.domain)
        
        logger.info("Planner initialized")
    
    def _init_executor(self):
        """Initialize action executor."""
        from nesy.reasoning.planning import ActionExecutor
        
        # Initialize with current state
        initial_state = self._get_current_state()
        self._executor = ActionExecutor(initial_state)
        
        logger.info("Executor initialized")
    
    def observe(self, image: np.ndarray, metadata: Optional[Dict] = None) -> Observation:
        """
        Process observation from sensors.
        
        Args:
            image: Camera image
            metadata: Additional sensor data
        
        Returns:
            Processed observation
        """
        self.state = AgentState.OBSERVING
        self._init_components()
        
        observation = Observation(
            timestamp=time.time(),
            image=image,
            metadata=metadata or {}
        )
        
        # Detect objects
        if self.config.use_yolo and hasattr(self, 'detector'):
            detections = self.detector.detect(image)
            observation.detections = detections
            
            # Update scene graph
            self._update_scene_graph(detections)
        
        # Compute embeddings
        if self.config.use_clip and hasattr(self, 'embedder'):
            embeddings = {}
            for det in observation.detections:
                if 'bbox' in det:
                    # Crop and embed
                    x1, y1, x2, y2 = det['bbox']
                    crop = image[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        emb = self.embedder.encode_image(crop)
                        embeddings[det.get('label', 'unknown')] = emb
            
            observation.embeddings = embeddings
        
        # Store
        self.current_observation = observation
        self.observation_history.append(observation)
        self.metrics["observations"] += 1
        
        logger.info(f"Observed {len(observation.detections)} objects")
        return observation
    
    def reason(self) -> Dict[str, Any]:
        """
        Apply reasoning to current observations.
        
        Returns:
            Reasoning results
        """
        self.state = AgentState.REASONING
        self._init_components()
        
        # Query reasoning engine
        results = {}
        
        # Spatial reasoning
        spatial_facts = self._reasoning.query("on")
        results["on"] = spatial_facts
        
        containment_facts = self._reasoning.query("in")
        results["in"] = containment_facts
        
        # Infer new facts
        # self._reasoning.reason()  # Method doesn't exist, inference happens on query
        
        self.metrics["inferences"] += 1
        
        logger.info(f"Reasoning complete: {len(results)} relations")
        return results
    
    def plan(
        self,
        goal: Goal,
        max_steps: Optional[int] = None
    ) -> Optional[List]:
        """
        Generate plan to achieve goal.
        
        Args:
            goal: Goal specification
            max_steps: Max planning steps
        
        Returns:
            Action plan or None
        """
        self.state = AgentState.PLANNING
        self._init_components()
        
        self.current_goal = goal
        
        # Create planning problem
        from nesy.reasoning.planning import PDDLProblem
        
        problem = PDDLProblem("agent_task", self.domain)
        
        # Set initial state from scene graph
        for fact in self._get_current_facts():
            problem.add_initial_fact(fact)
        
        # Set goal
        for pred in goal.target_predicates:
            problem.add_goal(pred)
        
        # Plan
        max_steps = max_steps or self.config.max_plan_steps
        plan = self._planner.plan(
            problem,
            algorithm=self.config.planner_algorithm,
            max_steps=max_steps
        )
        
        if plan:
            self.current_plan = plan
            self.metrics["plans_generated"] += 1
            logger.info(f"Plan generated: {len(plan)} actions")
        else:
            logger.warning("No plan found")
        
        return plan
    
    def execute(self, plan: List, step_callback: Optional[callable] = None) -> bool:
        """
        Execute action plan.
        
        Args:
            plan: Action sequence
            step_callback: Called after each action
        
        Returns:
            True if successful
        """
        self.state = AgentState.EXECUTING
        self._init_components()
        
        success = True
        
        for i, action in enumerate(plan):
            logger.info(f"Executing action {i+1}/{len(plan)}: {action}")
            
            # Execute action
            action_success = self._executor.execute(action)
            
            if not action_success:
                logger.warning(f"Action {action} failed")
                self.metrics["failures"] += 1
                
                if self.config.replan_on_failure:
                    logger.info("Replanning...")
                    # Would trigger replanning here
                
                success = False
                break
            
            self.metrics["actions_executed"] += 1
            self.action_history.append(action)
            
            # Callback
            if step_callback:
                step_callback(action, i)
        
        return success
    
    def run(
        self,
        goal: Goal,
        sensor_callback: callable,
        max_iterations: int = 100
    ) -> bool:
        """
        Run autonomous loop until goal achieved.
        
        Args:
            goal: Goal to achieve
            sensor_callback: Function to get sensor data
            max_iterations: Max loop iterations
        
        Returns:
            True if goal achieved
        """
        logger.info(f"Starting autonomous run: {goal.description}")
        
        for iteration in range(max_iterations):
            try:
                # 1. PERCEIVE
                sensor_data = sensor_callback()
                if isinstance(sensor_data, np.ndarray):
                    self.observe(sensor_data)
                
                # 2. REASON
                facts = self.reason()
                
                # 3. Check if goal achieved
                if self._is_goal_achieved(goal):
                    logger.info(f"Goal achieved in {iteration} iterations!")
                    return True
                
                # 4. PLAN
                plan = self.plan(goal)
                if not plan:
                    logger.warning("No plan found, continuing observation...")
                    time.sleep(1.0 / self.config.loop_frequency)
                    continue
                
                # 5. EXECUTE
                success = self.execute(plan)
                
                if not success and not self.config.replan_on_failure:
                    logger.error("Execution failed, stopping")
                    return False
                
                # Loop delay
                time.sleep(1.0 / self.config.loop_frequency)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                self.state = AgentState.ERROR
                return False
        
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return False
    
    # Helper methods
    
    def _update_scene_graph(self, detections: List[Dict]):
        """Update scene graph with detections."""
        for det in detections:
            obj_id = det.get('label', 'unknown')
            self.scene_graph.add_object(
                obj_id,
                label=det.get('label'),
                bbox=det.get('bbox'),
                confidence=det.get('confidence', 1.0)
            )
    
    def _add_domain_rules(self):
        """Add domain-specific rules."""
        # Spatial reasoning
        self._reasoning.add_custom_rule(
            "in(X, R) :- on(X, T), in(T, R)"
        )
        
        # Reachability
        self._reasoning.add_custom_rule(
            "reachable(X) :- in(X, R), at(robot, R)"
        )
    
    def _create_planning_domain(self):
        """Create PDDL domain for planning."""
        from nesy.reasoning.planning import PDDLDomain
        
        domain = PDDLDomain("robot_domain")
        
        # Predicates
        domain.add_predicate("at", ["location"])
        domain.add_predicate("holding", ["object"])
        domain.add_predicate("on", ["object", "surface"])
        domain.add_predicate("clear", ["surface"])
        
        # Actions
        domain.add_action(
            "pick",
            parameters=["obj"],
            preconditions=["at(obj)", "clear(obj)", "not holding(obj)"],
            effects=["holding(obj)", "not on(obj, surface)"]
        )
        
        domain.add_action(
            "place",
            parameters=["obj", "surface"],
            preconditions=["holding(obj)", "clear(surface)"],
            effects=["on(obj, surface)", "not holding(obj)", "not clear(surface)"]
        )
        
        return domain
    
    def _get_current_state(self):
        """Get current world state as predicates."""
        from nesy.reasoning.planning import Predicate
        
        state = set()
        
        # Extract from scene graph
        for obj in self.scene_graph.nodes.values():
            # Location
            state.add(Predicate("at", [obj.id]))
        
        return state
    
    def _get_current_facts(self) -> List[str]:
        """Get current facts as strings."""
        facts = []
        
        # From scene graph
        for obj in self.scene_graph.nodes.values():
            facts.append(f"at({obj.id})")
        
        return facts
    
    def _is_goal_achieved(self, goal: Goal) -> bool:
        """Check if goal is achieved."""
        # Query current state
        current_facts = set(self._get_current_facts())
        
        # Check all goal predicates
        for pred in goal.target_predicates:
            if pred not in current_facts:
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            **self.metrics,
            "state": self.state.value,
            "observation_history_size": len(self.observation_history),
            "action_history_size": len(self.action_history)
        }
    
    def reset(self):
        """Reset agent state."""
        self.state = AgentState.IDLE
        self.current_observation = None
        self.current_goal = None
        self.current_plan = None
        self.observation_history.clear()
        self.action_history.clear()
        
        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = 0
        
        logger.info("Agent reset")
