"""
Multi-Agent System.

Coordinates multiple autonomous agents for collaborative tasks.

Features:
- Agent registry and lifecycle management
- Task allocation and distribution
- Shared knowledge base
- Inter-agent communication
- Collaborative planning
- Conflict resolution

Example:
    >>> from nesy.agents.multi_agent import MultiAgentCoordinator
    >>> 
    >>> coord = MultiAgentCoordinator()
    >>> coord.register_agent("robot1", agent1)
    >>> coord.register_agent("robot2", agent2)
    >>> coord.assign_task("robot1", goal1)
    >>> coord.run_collaborative(shared_goal)
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import uuid

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent role in multi-agent system."""
    LEADER = "leader"
    WORKER = "worker"
    OBSERVER = "observer"
    PLANNER = "planner"


class MessageType(Enum):
    """Inter-agent message types."""
    REQUEST = "request"
    INFORM = "inform"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    BROADCAST = "broadcast"


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    receiver: str = ""  # Empty = broadcast
    msg_type: MessageType = MessageType.INFORM
    content: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0


@dataclass
class AgentInfo:
    """Registered agent information."""
    agent_id: str
    agent: Any  # AutonomousAgent instance
    role: AgentRole = AgentRole.WORKER
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    status: str = "idle"
    location: Optional[str] = None


@dataclass
class SharedKnowledge:
    """Shared knowledge base across agents."""
    facts: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict] = field(default_factory=list)
    plans: Dict[str, List] = field(default_factory=dict)
    conflicts: List[Dict] = field(default_factory=list)
    
    def add_fact(self, key: str, value: Any, source: str = ""):
        """Add fact to shared knowledge."""
        self.facts[key] = {
            "value": value,
            "source": source,
            "timestamp": time.time()
        }
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Get fact value."""
        entry = self.facts.get(key)
        return entry["value"] if entry else None
    
    def add_observation(self, obs: Dict):
        """Add shared observation."""
        obs["timestamp"] = time.time()
        self.observations.append(obs)
        # Keep last 1000
        if len(self.observations) > 1000:
            self.observations = self.observations[-1000:]


class MultiAgentCoordinator:
    """
    Coordinator for multi-agent systems.
    
    Manages agent registration, task allocation, communication,
    and collaborative planning.
    
    Example:
        >>> coord = MultiAgentCoordinator()
        >>> 
        >>> # Register agents
        >>> coord.register_agent("scout", agent1, role=AgentRole.OBSERVER)
        >>> coord.register_agent("worker", agent2, role=AgentRole.WORKER)
        >>> 
        >>> # Assign tasks
        >>> coord.assign_task("scout", "survey_room")
        >>> coord.assign_task("worker", "pick_objects")
        >>> 
        >>> # Collaborative planning
        >>> plan = coord.plan_collaborative(shared_goal)
    """
    
    def __init__(self):
        """Initialize coordinator."""
        self.agents: Dict[str, AgentInfo] = {}
        self.shared_kb = SharedKnowledge()
        self.message_queue: List[AgentMessage] = []
        self.message_history: List[AgentMessage] = []
        
        # Task management
        self.pending_tasks: List[Dict] = []
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: List[Dict] = []
        
        # Metrics
        self.metrics = {
            "messages_exchanged": 0,
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "conflicts_resolved": 0
        }
        
        logger.info("Multi-agent coordinator initialized")
    
    def register_agent(
        self,
        agent_id: str,
        agent: Any,
        role: AgentRole = AgentRole.WORKER,
        capabilities: Optional[List[str]] = None
    ):
        """
        Register an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
            role: Agent role
            capabilities: What the agent can do
        """
        info = AgentInfo(
            agent_id=agent_id,
            agent=agent,
            role=role,
            capabilities=capabilities or []
        )
        self.agents[agent_id] = info
        logger.info(f"Registered agent: {agent_id} (role={role.value})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent info."""
        return self.agents.get(agent_id)
    
    def get_agents_by_role(self, role: AgentRole) -> List[AgentInfo]:
        """Get all agents with a specific role."""
        return [a for a in self.agents.values() if a.role == role]
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get agents not currently assigned tasks."""
        return [a for a in self.agents.values() if a.status == "idle"]
    
    # --- Task Management ---
    
    def assign_task(
        self,
        agent_id: str,
        task_description: str,
        priority: float = 1.0
    ) -> bool:
        """
        Assign task to specific agent.
        
        Args:
            agent_id: Target agent
            task_description: Task to assign
            priority: Task priority
        
        Returns:
            True if assigned
        """
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f"Unknown agent: {agent_id}")
            return False
        
        task = {
            "id": str(uuid.uuid4())[:8],
            "description": task_description,
            "assigned_to": agent_id,
            "priority": priority,
            "status": "assigned",
            "timestamp": time.time()
        }
        
        agent.current_task = task["id"]
        agent.status = "busy"
        self.active_tasks[task["id"]] = task
        self.metrics["tasks_assigned"] += 1
        
        # Notify agent
        self.send_message(AgentMessage(
            sender="coordinator",
            receiver=agent_id,
            msg_type=MessageType.REQUEST,
            content={"task": task_description, "task_id": task["id"]}
        ))
        
        logger.info(f"Assigned task '{task_description}' to {agent_id}")
        return True
    
    def allocate_task(self, task_description: str, required_capabilities: Optional[List[str]] = None) -> Optional[str]:
        """
        Automatically allocate task to best available agent.
        
        Args:
            task_description: Task to allocate
            required_capabilities: Required agent capabilities
        
        Returns:
            Agent ID or None if no agent available
        """
        available = self.get_available_agents()
        
        if required_capabilities:
            # Filter by capabilities
            available = [
                a for a in available
                if all(cap in a.capabilities for cap in required_capabilities)
            ]
        
        if not available:
            logger.warning("No available agents for task")
            self.pending_tasks.append({
                "description": task_description,
                "requirements": required_capabilities or []
            })
            return None
        
        # Simple allocation: pick first available
        best = available[0]
        self.assign_task(best.agent_id, task_description)
        return best.agent_id
    
    def complete_task(self, task_id: str, result: Any = None):
        """Mark task as completed."""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task["status"] = "completed"
            task["result"] = result
            self.completed_tasks.append(task)
            
            # Free agent
            agent_id = task["assigned_to"]
            if agent_id in self.agents:
                self.agents[agent_id].status = "idle"
                self.agents[agent_id].current_task = None
            
            self.metrics["tasks_completed"] += 1
            logger.info(f"Task {task_id} completed by {agent_id}")
    
    # --- Communication ---
    
    def send_message(self, message: AgentMessage):
        """Send message to agent or broadcast."""
        self.message_queue.append(message)
        self.message_history.append(message)
        self.metrics["messages_exchanged"] += 1
        
        if not message.receiver:
            logger.debug(f"Broadcast from {message.sender}: {message.msg_type.value}")
        else:
            logger.debug(f"Message {message.sender} → {message.receiver}: {message.msg_type.value}")
    
    def get_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        messages = [
            m for m in self.message_queue
            if m.receiver == agent_id or m.receiver == ""
        ]
        # Remove delivered messages
        self.message_queue = [
            m for m in self.message_queue
            if m.receiver != agent_id and m.receiver != ""
        ]
        return messages
    
    def broadcast(self, sender: str, content: Dict):
        """Broadcast message to all agents."""
        self.send_message(AgentMessage(
            sender=sender,
            receiver="",
            msg_type=MessageType.BROADCAST,
            content=content
        ))
    
    # --- Shared Knowledge ---
    
    def share_observation(self, agent_id: str, observation: Dict):
        """Share observation with all agents."""
        observation["source"] = agent_id
        self.shared_kb.add_observation(observation)
        
        # Broadcast to others
        self.broadcast(agent_id, {
            "type": "observation",
            "data": observation
        })
    
    def share_fact(self, agent_id: str, key: str, value: Any):
        """Share fact."""
        self.shared_kb.add_fact(key, value, source=agent_id)
    
    # --- Collaborative Planning ---
    
    def plan_collaborative(
        self,
        goal_description: str,
        participating_agents: Optional[List[str]] = None
    ) -> Dict[str, List]:
        """
        Create collaborative plan.
        
        Decomposes goal and assigns sub-tasks to agents.
        
        Args:
            goal_description: Shared goal
            participating_agents: Which agents participate
        
        Returns:
            Dict mapping agent_id → action list
        """
        agents = participating_agents or list(self.agents.keys())
        
        if not agents:
            logger.warning("No agents for collaborative planning")
            return {}
        
        # Simple task decomposition (mock)
        plan = {}
        sub_tasks = self._decompose_goal(goal_description, len(agents))
        
        for i, agent_id in enumerate(agents):
            if i < len(sub_tasks):
                plan[agent_id] = sub_tasks[i]
                self.assign_task(agent_id, sub_tasks[i][0] if sub_tasks[i] else "wait")
        
        self.shared_kb.plans[goal_description] = plan
        logger.info(f"Collaborative plan created for {len(agents)} agents")
        
        return plan
    
    def _decompose_goal(self, goal: str, num_agents: int) -> List[List[str]]:
        """Decompose goal into sub-tasks per agent."""
        # Simple decomposition for demo
        sub_tasks = []
        
        if "clean" in goal.lower():
            tasks = ["survey_area", "collect_objects", "place_objects", "verify_clean"]
        elif "patrol" in goal.lower():
            tasks = [f"patrol_zone_{i+1}" for i in range(num_agents)]
        elif "search" in goal.lower():
            tasks = [f"search_area_{i+1}" for i in range(num_agents)]
        else:
            tasks = [f"subtask_{i+1}" for i in range(num_agents)]
        
        # Distribute tasks
        for i in range(num_agents):
            agent_tasks = [tasks[j] for j in range(len(tasks)) if j % num_agents == i]
            sub_tasks.append(agent_tasks)
        
        return sub_tasks
    
    # --- Conflict Resolution ---
    
    def detect_conflicts(self) -> List[Dict]:
        """Detect conflicts between agents."""
        conflicts = []
        
        agents_list = list(self.agents.values())
        for i, a1 in enumerate(agents_list):
            for a2 in agents_list[i+1:]:
                # Check location conflict
                if (a1.location and a2.location and
                    a1.location == a2.location and
                    a1.status == "busy" and a2.status == "busy"):
                    conflicts.append({
                        "type": "location",
                        "agents": [a1.agent_id, a2.agent_id],
                        "location": a1.location,
                        "resolution": "priority"
                    })
                
                # Check resource conflict (same task target)
                if (a1.current_task and a2.current_task and
                    a1.current_task == a2.current_task):
                    conflicts.append({
                        "type": "resource",
                        "agents": [a1.agent_id, a2.agent_id],
                        "task": a1.current_task,
                        "resolution": "reassign"
                    })
        
        self.shared_kb.conflicts = conflicts
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[Dict]):
        """Resolve detected conflicts."""
        for conflict in conflicts:
            if conflict["resolution"] == "priority":
                # Higher-role agent gets priority
                agents = conflict["agents"]
                a1 = self.agents.get(agents[0])
                a2 = self.agents.get(agents[1])
                
                if a1 and a2:
                    # Leader always wins
                    if a2.role == AgentRole.LEADER:
                        a1, a2 = a2, a1
                    
                    logger.info(f"Conflict resolved: {a1.agent_id} prioritized over {a2.agent_id}")
            
            self.metrics["conflicts_resolved"] += 1
    
    # --- Execution ---
    
    def step(self):
        """Execute one coordination step."""
        # Check for conflicts
        conflicts = self.detect_conflicts()
        if conflicts:
            self.resolve_conflicts(conflicts)
        
        # Process pending tasks
        while self.pending_tasks and self.get_available_agents():
            task = self.pending_tasks.pop(0)
            self.allocate_task(task["description"], task.get("requirements"))
    
    def get_status(self) -> Dict:
        """Get coordinator status."""
        return {
            "agents": {
                aid: {"role": a.role.value, "status": a.status, "task": a.current_task}
                for aid, a in self.agents.items()
            },
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "shared_facts": len(self.shared_kb.facts),
            "metrics": self.metrics
        }
