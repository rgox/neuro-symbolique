"""
Tests for Production API.

Tests FastAPI endpoints, metrics, and application state.
"""

import pytest
from unittest.mock import Mock

# Try importing test client
try:
    from fastapi.testclient import TestClient
    from nesy.api.server import create_app, AppState
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

skipif_no_fastapi = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed"
)


@skipif_no_fastapi
class TestAppState:
    """Test application state."""
    
    def test_create_state(self):
        state = AppState()
        assert state.request_count == 0
        assert state.uptime > 0
    
    def test_avg_response_time_zero(self):
        state = AppState()
        assert state.avg_response_time == 0.0
    
    def test_avg_response_time(self):
        state = AppState()
        state.request_count = 2
        state.total_response_time = 100.0
        assert state.avg_response_time == 50.0
    
    def test_init_components(self):
        state = AppState()
        state.init_components()
        assert state._initialized
        assert state.scene_graph is not None
        assert state.reasoning_engine is not None
        assert state.agent is not None


@skipif_no_fastapi
class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data


@skipif_no_fastapi
class TestMetricsEndpoint:
    """Test metrics endpoints."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_prometheus_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        assert "nesy_uptime_seconds" in text
        assert "nesy_requests_total" in text
    
    def test_api_metrics(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "requests_total" in data


@skipif_no_fastapi
class TestSceneEndpoints:
    """Test scene graph endpoints."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_list_objects(self, client):
        response = client.get("/api/v1/scene/objects")
        assert response.status_code == 200
        data = response.json()
        assert "objects" in data
        assert "count" in data
    
    def test_add_object(self, client):
        response = client.post("/api/v1/scene/objects", json={
            "object_id": "cup_1",
            "label": "cup",
            "position": [1.0, 2.0, 0.5],
            "attributes": {"color": "red"}
        })
        assert response.status_code == 200
        assert response.json()["status"] == "created"


@skipif_no_fastapi
class TestReasoningEndpoints:
    """Test reasoning endpoints."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_query(self, client):
        response = client.post("/api/v1/reason", json={
            "query": "on",
            "method": "scallop"
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "inference_time_ms" in data


@skipif_no_fastapi
class TestAgentEndpoints:
    """Test agent control endpoints."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_agent_status(self, client):
        response = client.post("/api/v1/agent/command", json={
            "command": "status"
        })
        assert response.status_code == 200
        assert "state" in response.json()
    
    def test_agent_reset(self, client):
        response = client.post("/api/v1/agent/command", json={
            "command": "reset"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "reset"
    
    def test_agent_unknown_command(self, client):
        response = client.post("/api/v1/agent/command", json={
            "command": "unknown"
        })
        assert response.status_code == 400


@skipif_no_fastapi
class TestDocs:
    """Test API documentation."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_openapi(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_docs(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
