"""
FastAPI Server for Neuro-Symbolic AI Platform.

REST API for perception, reasoning, planning, and agent control.

Endpoints:
    /health          - Health check
    /metrics         - Prometheus metrics
    /api/v1/scene    - Scene graph operations
    /api/v1/reason   - Reasoning queries
    /api/v1/plan     - Planning operations
    /api/v1/agent    - Agent control

Example:
    uvicorn nesy.api.server:app --host 0.0.0.0 --port 8000
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

# Try importing FastAPI
try:
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.info("FastAPI not installed. Install with: pip install fastapi uvicorn")


# --- Pydantic Models ---

if FASTAPI_AVAILABLE:
    class HealthResponse(BaseModel):
        status: str
        version: str
        uptime: float
        components: Dict[str, str]

    class SceneObjectRequest(BaseModel):
        object_id: str
        label: str
        position: List[float] = [0.0, 0.0, 0.0]
        attributes: Dict[str, Any] = {}

    class ReasoningQueryRequest(BaseModel):
        query: str
        method: str = "scallop"  # scallop, vsa, hybrid

    class ReasoningResponse(BaseModel):
        query: str
        results: List[Dict[str, Any]]
        method: str
        inference_time_ms: float

    class PlanRequest(BaseModel):
        goal: str
        target_predicates: List[str] = []
        algorithm: str = "astar"

    class PlanResponse(BaseModel):
        goal: str
        plan: List[str]
        num_steps: int
        planning_time_ms: float

    class AgentCommandRequest(BaseModel):
        command: str  # start, stop, reset, observe, reason, plan
        params: Dict[str, Any] = {}

    class MetricsResponse(BaseModel):
        observations: int = 0
        inferences: int = 0
        plans_generated: int = 0
        actions_executed: int = 0
        uptime_seconds: float = 0.0
        requests_total: int = 0
        avg_response_time_ms: float = 0.0


# --- Application State ---

class AppState:
    """Application state manager."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0.0
        self.scene_graph = None
        self.reasoning_engine = None
        self.agent = None
        self._initialized = False
    
    def init_components(self):
        """Lazy initialize components."""
        if self._initialized:
            return
        
        try:
            from nesy.world_model.scene_graph import SceneGraph
            self.scene_graph = SceneGraph()
            
            from nesy.reasoning.logic import ReasoningEngine
            self.reasoning_engine = ReasoningEngine(self.scene_graph)
            
            from nesy.agents import AutonomousAgent, AgentConfig
            config = AgentConfig(use_yolo=False, use_clip=False)
            self.agent = AutonomousAgent(config)
            
            self._initialized = True
            logger.info("API components initialized")
        except Exception as e:
            logger.error(f"Component init error: {e}")
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_response_time(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_response_time / self.request_count


# --- Create App ---

def create_app() -> 'FastAPI':
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed")
    
    app = FastAPI(
        title="Neuro-Symbolic AI Platform API",
        description="REST API for perception, reasoning, planning & agent control",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    state = AppState()
    
    # --- Health ---
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=state.uptime,
            components={
                "scene_graph": "ok" if state.scene_graph else "not_initialized",
                "reasoning": "ok" if state.reasoning_engine else "not_initialized",
                "agent": "ok" if state.agent else "not_initialized",
            }
        )
    
    # --- Metrics ---
    
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus-compatible metrics."""
        metrics = []
        metrics.append(f'nesy_uptime_seconds {state.uptime:.2f}')
        metrics.append(f'nesy_requests_total {state.request_count}')
        metrics.append(f'nesy_avg_response_time_ms {state.avg_response_time:.2f}')
        
        if state.agent:
            agent_metrics = state.agent.get_metrics()
            metrics.append(f'nesy_observations_total {agent_metrics["observations"]}')
            metrics.append(f'nesy_inferences_total {agent_metrics["inferences"]}')
            metrics.append(f'nesy_plans_total {agent_metrics["plans_generated"]}')
            metrics.append(f'nesy_actions_total {agent_metrics["actions_executed"]}')
        
        return "\n".join(metrics) + "\n"
    
    @app.get("/api/v1/metrics", response_model=MetricsResponse)
    async def get_metrics():
        state.init_components()
        agent_metrics = state.agent.get_metrics() if state.agent else {}
        return MetricsResponse(
            observations=agent_metrics.get("observations", 0),
            inferences=agent_metrics.get("inferences", 0),
            plans_generated=agent_metrics.get("plans_generated", 0),
            actions_executed=agent_metrics.get("actions_executed", 0),
            uptime_seconds=state.uptime,
            requests_total=state.request_count,
            avg_response_time_ms=state.avg_response_time
        )
    
    # --- Scene Graph ---
    
    @app.get("/api/v1/scene/objects")
    async def list_objects():
        state.init_components()
        state.request_count += 1
        
        objects = []
        for node in state.scene_graph.nodes.values():
            objects.append({
                "id": node.id,
                "attributes": node.attributes,
                "position": node.position.tolist(),
                "confidence": node.confidence
            })
        return {"objects": objects, "count": len(objects)}
    
    @app.post("/api/v1/scene/objects")
    async def add_object(req: SceneObjectRequest):
        state.init_components()
        state.request_count += 1
        start = time.time()
        
        from nesy.world_model.scene_graph.graph import Node, LayerType, NodeType
        import numpy as np
        pos = np.array(req.position, dtype=np.float32) if req.position else np.zeros(3)
        state.scene_graph.add_node(
            layer=LayerType.L1,
            node_type=NodeType.OBJECT,
            position=pos,
            attributes={"label": req.label, **req.attributes},
            node_id=req.object_id
        )
        
        elapsed = (time.time() - start) * 1000
        state.total_response_time += elapsed
        
        return {"status": "created", "object_id": req.object_id, "time_ms": elapsed}
    
    @app.delete("/api/v1/scene/objects/{object_id}")
    async def remove_object(object_id: str):
        state.init_components()
        state.request_count += 1
        
        try:
            state.scene_graph.remove_node(object_id)
            return {"status": "deleted", "object_id": object_id}
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    # --- Reasoning ---
    
    @app.post("/api/v1/reason", response_model=ReasoningResponse)
    async def query_reasoning(req: ReasoningQueryRequest):
        state.init_components()
        state.request_count += 1
        start = time.time()
        
        results = state.reasoning_engine.query(req.query)
        
        elapsed = (time.time() - start) * 1000
        state.total_response_time += elapsed
        
        return ReasoningResponse(
            query=req.query,
            results=[{"fact": str(r)} for r in results],
            method=req.method,
            inference_time_ms=elapsed
        )
    
    @app.post("/api/v1/reason/rule")
    async def add_rule(rule: Dict[str, str]):
        state.init_components()
        state.request_count += 1
        
        rule_text = rule.get("rule", "")
        state.reasoning_engine.add_custom_rule(rule_text)
        
        return {"status": "added", "rule": rule_text}
    
    # --- Planning ---
    
    @app.post("/api/v1/plan", response_model=PlanResponse)
    async def create_plan(req: PlanRequest):
        state.init_components()
        state.request_count += 1
        start = time.time()
        
        from nesy.agents import Goal
        goal = Goal(
            description=req.goal,
            target_predicates=req.target_predicates
        )
        
        plan = state.agent.plan(goal)
        
        elapsed = (time.time() - start) * 1000
        state.total_response_time += elapsed
        
        plan_strs = [str(a) for a in plan] if plan else []
        
        return PlanResponse(
            goal=req.goal,
            plan=plan_strs,
            num_steps=len(plan_strs),
            planning_time_ms=elapsed
        )
    
    # --- Agent Control ---
    
    @app.post("/api/v1/agent/command")
    async def agent_command(req: AgentCommandRequest):
        state.init_components()
        state.request_count += 1
        
        if req.command == "reset":
            state.agent.reset()
            return {"status": "reset", "state": state.agent.state.value}
        
        elif req.command == "status":
            return {
                "state": state.agent.state.value,
                "metrics": state.agent.get_metrics()
            }
        
        elif req.command == "reason":
            results = state.agent.reason()
            return {"status": "ok", "results": {k: [str(v) for v in vals] for k, vals in results.items()}}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {req.command}")
    
    # --- WebSocket for streaming ---
    
    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                
                command = data.get("command", "status")
                
                if command == "status":
                    metrics = state.agent.get_metrics() if state.agent else {}
                    await websocket.send_json({
                        "type": "status",
                        "data": metrics
                    })
                elif command == "subscribe_metrics":
                    # Send metrics updates
                    if state.agent:
                        await websocket.send_json({
                            "type": "metrics",
                            "data": state.agent.get_metrics()
                        })
                
        except Exception:
            pass
    
    return app


# Create default app instance
if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
