"""Final v1 coverage push — targets all modules below 85%."""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════
# API Server (74%) — FastAPI TestClient
# ═══════════════════════════════════════════════════════════════════

class TestAPIServerRoutes:
    @pytest.fixture
    def client(self):
        from nesy.api.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "uptime" in data

    def test_prometheus_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "nesy_uptime_seconds" in r.text

    def test_api_metrics(self, client):
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "uptime_seconds" in data

    def test_list_objects_empty(self, client):
        r = client.get("/api/v1/scene/objects")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_add_object(self, client):
        r = client.post("/api/v1/scene/objects", json={
            "object_id": "cup_1",
            "label": "cup",
            "position": [1.0, 2.0, 0.5],
            "attributes": {"color": "red"},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "created"

    def test_add_then_list(self, client):
        client.post("/api/v1/scene/objects", json={
            "object_id": "obj1", "label": "table",
        })
        r = client.get("/api/v1/scene/objects")
        assert r.json()["count"] >= 1

    def test_delete_object(self, client):
        # Add then delete
        client.post("/api/v1/scene/objects", json={
            "object_id": "del_test", "label": "cup",
        })
        r = client.delete("/api/v1/scene/objects/del_test")
        assert r.status_code == 200

    def test_reasoning_query(self, client):
        r = client.post("/api/v1/reason", json={"query": "on", "method": "scallop"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "inference_time_ms" in data

    def test_add_rule(self, client):
        r = client.post("/api/v1/reason/rule", json={"rule": "path(X,Y) :- edge(X,Y)"})
        assert r.status_code == 200
        assert r.json()["status"] == "added"

    def test_create_plan(self, client):
        r = client.post("/api/v1/plan", json={
            "goal": "find cup",
            "target_predicates": ["found(cup)"],
        })
        assert r.status_code == 200
        data = r.json()
        assert "plan" in data
        assert "planning_time_ms" in data

    def test_agent_reset(self, client):
        r = client.post("/api/v1/agent/command", json={"command": "reset"})
        assert r.status_code == 200
        assert r.json()["status"] == "reset"

    def test_agent_status(self, client):
        r = client.post("/api/v1/agent/command", json={"command": "status"})
        assert r.status_code == 200
        assert "state" in r.json()

    def test_agent_reason(self, client):
        r = client.post("/api/v1/agent/command", json={"command": "reason"})
        assert r.status_code == 200

    def test_agent_unknown_command(self, client):
        r = client.post("/api/v1/agent/command", json={"command": "fly"})
        assert r.status_code == 400


class TestAppState:
    def test_uptime(self):
        from nesy.api.server import AppState
        state = AppState()
        time.sleep(0.01)
        assert state.uptime > 0

    def test_avg_response_time_zero(self):
        from nesy.api.server import AppState
        state = AppState()
        assert state.avg_response_time == 0.0

    def test_avg_response_time(self):
        from nesy.api.server import AppState
        state = AppState()
        state.request_count = 2
        state.total_response_time = 100.0
        assert state.avg_response_time == 50.0

    def test_init_components(self):
        from nesy.api.server import AppState
        state = AppState()
        state.init_components()
        assert state._initialized
        assert state.scene_graph is not None
        # Second call should be no-op
        state.init_components()


# ═══════════════════════════════════════════════════════════════════
# HyperVector (80%) — full VSA operations
# ═══════════════════════════════════════════════════════════════════

class TestHyperVectorOps:
    def test_random_real(self):
        from nesy.reasoning.vsa.hypervector import HyperVector, HyperVectorType
        hv = HyperVector.random(1000, HyperVectorType.REAL, seed=42)
        assert hv.dim == 1000
        assert abs(np.linalg.norm(hv.data) - 1.0) < 0.01

    def test_random_binary(self):
        from nesy.reasoning.vsa.hypervector import HyperVector, HyperVectorType
        hv = HyperVector.random(1000, HyperVectorType.BINARY, seed=42)
        assert hv.dim == 1000
        assert set(np.unique(hv.data)).issubset({-1.0, 1.0})

    def test_zero(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        hv = HyperVector.zero(500)
        assert np.all(hv.data == 0)

    def test_from_vector(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        v = np.array([3.0, 4.0], dtype=np.float32)
        hv = HyperVector.from_vector(v)
        assert abs(np.linalg.norm(hv.data) - 1.0) < 0.01

    def test_bind_unbind(self):
        from nesy.reasoning.vsa.hypervector import HyperVector, HyperVectorType
        a = HyperVector.random(1000, HyperVectorType.BINARY, seed=1)
        b = HyperVector.random(1000, HyperVectorType.BINARY, seed=2)
        bound = a.bind(b)
        # Unbind: (a * b) * b = a (for binary)
        unbound = bound.bind(b)
        sim = a.similarity(unbound)
        assert sim > 0.9

    def test_bind_real(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        a = HyperVector.random(1000, seed=1)
        b = HyperVector.random(1000, seed=2)
        bound = a.bind(b)
        assert bound.dim == 1000

    def test_bundle(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        a = HyperVector.random(1000, seed=1)
        b = HyperVector.random(1000, seed=2)
        bundled = a.bundle(b)
        # Bundled should be similar to both
        assert a.similarity(bundled) > 0.3
        assert b.similarity(bundled) > 0.3

    def test_permute(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        hv = HyperVector.random(1000, seed=1)
        p = hv.permute(5)
        # Shifted should be dissimilar
        assert abs(hv.similarity(p)) < 0.3

    def test_similarity_self(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        hv = HyperVector.random(1000, seed=1)
        assert abs(hv.similarity(hv) - 1.0) < 0.01

    def test_threshold(self):
        from nesy.reasoning.vsa.hypervector import HyperVector, HyperVectorType
        hv = HyperVector.random(1000, seed=1)
        binary = hv.threshold(0.0)
        assert binary.hv_type == HyperVectorType.BINARY

    def test_add_operator(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        a = HyperVector.random(500, seed=1)
        b = HyperVector.random(500, seed=2)
        c = a + b
        assert c.dim == 500

    def test_mul_bind(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        a = HyperVector.random(500, seed=1)
        b = HyperVector.random(500, seed=2)
        c = a * b
        assert c.dim == 500

    def test_mul_scalar(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        a = HyperVector.random(500, seed=1)
        c = a * 2.0
        assert c.dim == 500

    def test_repr(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        hv = HyperVector.random(100, seed=1)
        r = repr(hv)
        assert "dim=100" in r

    def test_to_from_dict(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        hv = HyperVector.random(100, seed=42)
        d = hv.to_dict()
        hv2 = HyperVector.from_dict(d)
        assert hv.similarity(hv2) > 0.99

    def test_normalize_already(self):
        from nesy.reasoning.vsa.hypervector import HyperVector
        hv = HyperVector.random(100, seed=1)
        n = hv.normalize()
        assert n is hv  # Already normalized

    def test_cosine_batch(self):
        from nesy.reasoning.vsa.hypervector import HyperVector, cosine_similarity_batch
        q = HyperVector.random(500, seed=1)
        hvs = [HyperVector.random(500, seed=i) for i in range(5)]
        sims = cosine_similarity_batch(q, hvs)
        assert len(sims) == 5
        # Self similarity should be highest
        sims_with_self = cosine_similarity_batch(q, [q] + hvs)
        assert sims_with_self[0] > 0.99


# ═══════════════════════════════════════════════════════════════════
# VSA Codebook (84%) — encode, decode, attributes, serialization
# ═══════════════════════════════════════════════════════════════════

class TestVSACodebook:
    def test_encode_deterministic(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        hv1 = cb.encode("cup")
        hv2 = cb.encode("cup")
        assert abs(hv1.similarity(hv2) - 1.0) < 1e-5

    def test_encode_different_symbols(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        hv1 = cb.encode("cup")
        hv2 = cb.encode("table")
        assert abs(hv1.similarity(hv2)) < 0.3

    def test_decode(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=1000, seed=42)
        cb.encode("cup")
        cb.encode("table")
        cb.encode("chair")
        hv = cb.encode("cup")
        results = cb.decode(hv, top_k=3)
        assert results[0][0] == "cup"
        assert results[0][1] > 0.9

    def test_decode_empty(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500)
        from nesy.reasoning.vsa.hypervector import HyperVector
        results = cb.decode(HyperVector.random(500))
        assert results == []

    def test_bind_attribute(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        hv = cb.bind_attribute("COLOR", "red")
        assert hv.dim == 500

    def test_bind_attribute_hv_value(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        from nesy.reasoning.vsa.hypervector import HyperVector
        cb = VSACodebook(dim=500, seed=42)
        hv_val = HyperVector.random(500, seed=99)
        hv = cb.bind_attribute("KEY", hv_val, encode_value=True)
        assert hv.dim == 500

    def test_bind_attribute_raw_hv(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        from nesy.reasoning.vsa.hypervector import HyperVector
        cb = VSACodebook(dim=500, seed=42)
        hv_val = HyperVector.random(500, seed=99)
        hv = cb.bind_attribute("KEY", hv_val, encode_value=False)
        assert hv.dim == 500

    def test_encode_position(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        pos = np.array([1.5, 2.0, 0.5])
        hv = cb.encode_position(pos)
        assert hv.dim == 500

    def test_encode_attributes(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        hv = cb.encode_attributes({"color": "red", "size": "large"})
        assert hv.dim == 500

    def test_encode_attributes_empty(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        hv = cb.encode_attributes({})
        assert np.all(hv.data == 0)  # Zero vector for no attributes

    def test_clear(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        cb.encode("a")
        cb.encode("b")
        cb.clear()
        assert cb.size() == 0
        assert cb.get_all_symbols() == []

    def test_size_and_symbols(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        cb.encode("x")
        cb.encode("y")
        assert cb.size() == 2
        assert set(cb.get_all_symbols()) == {"x", "y"}

    def test_repr(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        r = repr(cb)
        assert "500" in r

    def test_to_from_dict(self):
        from nesy.reasoning.vsa.codebook import VSACodebook
        cb = VSACodebook(dim=500, seed=42)
        cb.encode("cup")
        cb.encode("table")
        d = cb.to_dict()
        cb2 = VSACodebook.from_dict(d)
        assert cb2.size() == 2
        # Same symbols should produce same hypervectors
        sim = cb.encode("cup").similarity(cb2.encode("cup"))
        assert sim > 0.99


# ═══════════════════════════════════════════════════════════════════
# ModelZoo (78%) — registry, caching, device resolution
# ═══════════════════════════════════════════════════════════════════

class TestModelZoo:
    def setup_method(self):
        from nesy.perception.models.zoo import ModelZoo
        self._orig_registry = dict(ModelZoo._registry)
        self._orig_cache = dict(ModelZoo._cache)

    def teardown_method(self):
        from nesy.perception.models.zoo import ModelZoo
        ModelZoo._registry.clear()
        ModelZoo._registry.update(self._orig_registry)
        ModelZoo._cache.clear()
        ModelZoo._cache.update(self._orig_cache)

    def test_register_and_get(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        ModelZoo.register("test_model_unique", ModelConfig(
            name="test_model_unique",
            model_type="detector",
            loader=lambda **kw: "loaded_model",
            device="cpu",
        ))
        m = ModelZoo.get("test_model_unique")
        assert m == "loaded_model"

    def test_get_cached(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        call_count = [0]
        def loader(**kw):
            call_count[0] += 1
            return "model"
        ModelZoo.register("cached_test", ModelConfig(
            name="cached_test", model_type="detector", loader=loader, device="cpu",
        ))
        ModelZoo.get("cached_test")
        ModelZoo.get("cached_test")
        assert call_count[0] == 1  # Only loaded once

    def test_get_force_reload(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        call_count = [0]
        def loader(**kw):
            call_count[0] += 1
            return "model"
        ModelZoo.register("reload_test", ModelConfig(
            name="reload_test", model_type="detector", loader=loader, device="cpu",
        ))
        ModelZoo.get("reload_test")
        ModelZoo.get("reload_test", force_reload=True)
        assert call_count[0] == 2

    def test_get_not_registered(self):
        from nesy.perception.models.zoo import ModelZoo
        with pytest.raises(KeyError):
            ModelZoo.get("nonexistent_model_xyz")

    def test_get_detector(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        ModelZoo.register("det_test", ModelConfig(
            name="det_test", model_type="detector",
            loader=lambda **kw: "det", device="cpu",
        ))
        assert ModelZoo.get_detector("det_test") == "det"

    def test_get_detector_wrong_type(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        ModelZoo.register("emb_test", ModelConfig(
            name="emb_test", model_type="embedder",
            loader=lambda **kw: "emb", device="cpu",
        ))
        with pytest.raises(ValueError):
            ModelZoo.get_detector("emb_test")

    def test_get_embedding_model(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        ModelZoo.register("emb_ok", ModelConfig(
            name="emb_ok", model_type="embedder",
            loader=lambda **kw: "emb", device="cpu",
        ))
        assert ModelZoo.get_embedding_model("emb_ok") == "emb"

    def test_get_embedding_wrong_type(self):
        from nesy.perception.models.zoo import ModelZoo, ModelConfig
        ModelZoo.register("det_wrong", ModelConfig(
            name="det_wrong", model_type="detector",
            loader=lambda **kw: "det", device="cpu",
        ))
        with pytest.raises(ValueError):
            ModelZoo.get_embedding_model("det_wrong")

    def test_list_models(self):
        from nesy.perception.models.zoo import ModelZoo
        all_models = ModelZoo.list_models()
        assert isinstance(all_models, list)
        assert len(all_models) > 0  # Pre-registered YOLO/CLIP

    def test_list_models_by_type(self):
        from nesy.perception.models.zoo import ModelZoo
        detectors = ModelZoo.list_models("detector")
        assert len(detectors) > 0

    def test_clear_cache_all(self):
        from nesy.perception.models.zoo import ModelZoo
        ModelZoo._cache["test"] = "val"
        ModelZoo.clear_cache()
        assert "test" not in ModelZoo._cache

    def test_clear_cache_specific(self):
        from nesy.perception.models.zoo import ModelZoo
        ModelZoo._cache["a"] = 1
        ModelZoo._cache["b"] = 2
        ModelZoo.clear_cache("a")
        assert "a" not in ModelZoo._cache
        assert "b" in ModelZoo._cache

    def test_set_default_device(self):
        from nesy.perception.models.zoo import ModelZoo
        ModelZoo.set_default_device("cpu")
        assert ModelZoo._default_device == "cpu"

    def test_get_device_auto(self):
        from nesy.perception.models.zoo import ModelZoo
        device = ModelZoo._get_device("auto")
        assert device in ("cpu", "cuda")

    def test_get_device_explicit(self):
        from nesy.perception.models.zoo import ModelZoo
        assert ModelZoo._get_device("cpu") == "cpu"


# ═══════════════════════════════════════════════════════════════════
# Saliency (75%) — spectral, edge, contrast methods
# ═══════════════════════════════════════════════════════════════════

cv2 = pytest.importorskip("cv2")

class TestSaliencyDetector:
    def _make_image(self, h=100, w=150):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Add a bright spot for saliency
        img[30:50, 60:90] = 255
        return img

    def test_spectral(self):
        from nesy.perception.attention.saliency import SaliencyDetector, SaliencyMethod
        det = SaliencyDetector(method=SaliencyMethod.SPECTRAL)
        img = self._make_image()
        sal = det.compute_saliency(img)
        assert sal.shape == (100, 150)

    def test_edges(self):
        from nesy.perception.attention.saliency import SaliencyDetector, SaliencyMethod
        det = SaliencyDetector(method=SaliencyMethod.EDGES)
        sal = det.compute_saliency(self._make_image())
        assert sal.shape[:2] == (100, 150)

    def test_contrast(self):
        from nesy.perception.attention.saliency import SaliencyDetector, SaliencyMethod
        det = SaliencyDetector(method=SaliencyMethod.CONTRAST)
        sal = det.compute_saliency(self._make_image())
        assert sal.shape[:2] == (100, 150)

    def test_fine_grained_fallback(self):
        from nesy.perception.attention.saliency import SaliencyDetector, SaliencyMethod
        det = SaliencyDetector(method=SaliencyMethod.FINE_GRAINED)
        sal = det.compute_saliency(self._make_image())
        assert sal.shape[:2] == (100, 150)

    def test_grayscale_input(self):
        from nesy.perception.attention.saliency import SaliencyDetector, SaliencyMethod
        det = SaliencyDetector(method=SaliencyMethod.SPECTRAL)
        gray = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        sal = det.compute_saliency(gray)
        assert sal.shape == (100, 150)

    def test_get_top_points(self):
        from nesy.perception.attention.saliency import SaliencyDetector
        det = SaliencyDetector(threshold=50)
        img = self._make_image()
        sal = det.compute_saliency(img)
        points = det.get_top_points(sal, k=3)
        assert isinstance(points, list)

    def test_get_salient_regions(self):
        from nesy.perception.attention.saliency import SaliencyDetector
        det = SaliencyDetector(threshold=50)
        img = self._make_image()
        sal = det.compute_saliency(img)
        regions = det.get_salient_regions(sal, min_area=10)
        assert isinstance(regions, list)

    def test_visualize(self):
        from nesy.perception.attention.saliency import SaliencyDetector
        det = SaliencyDetector()
        img = self._make_image()
        sal = det.compute_saliency(img)
        viz = det.visualize(img, sal)
        assert viz.shape == img.shape

    def test_no_blur(self):
        from nesy.perception.attention.saliency import SaliencyDetector
        det = SaliencyDetector(gaussian_blur=0)
        sal = det.compute_saliency(self._make_image())
        assert sal.shape[:2] == (100, 150)


# ═══════════════════════════════════════════════════════════════════
# NLQuery (80%) — more patterns
# ═══════════════════════════════════════════════════════════════════

class TestNLQueryExtra:
    def _make_engine_and_sg(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
        sg = SceneGraph()
        cup = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1], attributes={"class": "cup", "color": "red"})
        table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 0], attributes={"class": "table"})
        sg.add_edge(cup.id, table.id, RelationType.ON)
        engine = ReasoningEngine(scene_graph=sg)
        engine.sync_from_scene_graph()
        return engine, sg

    def test_what_is_on(self):
        from nesy.reasoning.nlquery import NLQueryInterface
        engine, sg = self._make_engine_and_sg()
        nli = NLQueryInterface(engine, sg)
        r = nli.query("what is on table")
        assert r is not None

    def test_how_many(self):
        from nesy.reasoning.nlquery import NLQueryInterface
        engine, sg = self._make_engine_and_sg()
        nli = NLQueryInterface(engine, sg)
        r = nli.query("how many objects")
        assert r is not None

    def test_list_all(self):
        from nesy.reasoning.nlquery import NLQueryInterface
        engine, sg = self._make_engine_and_sg()
        nli = NLQueryInterface(engine, sg)
        r = nli.query("list all objects")
        assert r is not None

    def test_find_pattern(self):
        from nesy.reasoning.nlquery import NLQueryInterface
        engine, sg = self._make_engine_and_sg()
        nli = NLQueryInterface(engine, sg)
        r = nli.query("find cup")
        assert r is not None

    def test_unknown_query(self):
        from nesy.reasoning.nlquery import NLQueryInterface
        engine, sg = self._make_engine_and_sg()
        nli = NLQueryInterface(engine, sg)
        r = nli.query("xyzzy random nonsense query")
        # Should not crash, returns something


# ═══════════════════════════════════════════════════════════════════
# Advanced Reasoning (83%) — aggregation, probabilistic, validation
# ═══════════════════════════════════════════════════════════════════

class TestAdvancedReasoningExtra:
    def _make_engine(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
        sg = SceneGraph()
        cup = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1], attributes={"class": "cup"})
        mug = sg.add_node(LayerType.L1, NodeType.OBJECT, [2, 1, 1], attributes={"class": "mug"})
        table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 0], attributes={"class": "table"})
        sg.add_edge(cup.id, table.id, RelationType.ON)
        sg.add_edge(mug.id, table.id, RelationType.ON)
        engine = ReasoningEngine(scene_graph=sg)
        engine.sync_from_scene_graph()
        return engine, sg

    def test_aggregation_count_objects_in(self):
        from nesy.reasoning.logic.advanced import AggregationRules
        engine, sg = self._make_engine()
        agg = AggregationRules(engine)
        count = agg.count_objects_in("table")
        assert isinstance(count, int)

    def test_aggregation_count_by_type(self):
        from nesy.reasoning.logic.advanced import AggregationRules
        engine, sg = self._make_engine()
        agg = AggregationRules(engine)
        counts = agg.count_by_type()
        assert isinstance(counts, dict)

    def test_probabilistic_add_and_query(self):
        from nesy.reasoning.logic.advanced import ProbabilisticRules
        engine, sg = self._make_engine()
        prob = ProbabilisticRules(engine)
        prob.add_probabilistic_fact("on", "cup", "table", confidence=0.9)
        result = prob.query_with_confidence("on", "cup", "table")
        assert isinstance(result, tuple)

    def test_rule_validator(self):
        from nesy.reasoning.logic.advanced import RuleValidator
        engine, sg = self._make_engine()
        rv = RuleValidator(engine)
        valid = rv.validate_rule("path(X,Y) :- edge(X,Y)")
        assert isinstance(valid, bool)


# ═══════════════════════════════════════════════════════════════════
# Device (82%) — base class, telemetry, statistics
# ═══════════════════════════════════════════════════════════════════

class TestDeviceBase:
    def test_device_statistics(self):
        from nesy.hal.npu.simulator import NPU
        from nesy.hal.device import Workload, WorkloadType, ExecutionResult
        from nesy.core.memory import UMA
        npu = NPU(uma=UMA())
        stats = npu.get_statistics()
        assert isinstance(stats, dict)
        assert "total_executions" in stats

    def test_device_pool_statistics(self):
        from nesy.hal.device import DevicePool
        from nesy.hal.npu.simulator import NPU
        from nesy.core.memory import UMA
        uma = UMA()
        pool = DevicePool(uma=uma)
        pool.register_device(NPU(uma=uma))
        stats = pool.get_all_statistics()
        assert isinstance(stats, dict)

    def test_device_pool_find_best_match(self):
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.hal.npu.simulator import NPU
        from nesy.hal.spu.simulator import SPU
        from nesy.core.memory import UMA
        uma = UMA()
        pool = DevicePool(uma=uma)
        pool.register_device(NPU(uma=uma))
        pool.register_device(SPU(uma=uma))

        wl_neural = Workload(WorkloadType.NEURAL_INFERENCE, "model", {})
        dev = pool.find_device_for_workload(wl_neural)
        assert dev is not None

        wl_graph = Workload(WorkloadType.GRAPH_TRAVERSAL, "bfs", {})
        dev = pool.find_device_for_workload(wl_graph)
        assert dev is not None


# ═══════════════════════════════════════════════════════════════════
# ReasoningEngine (84%) — query, inference extras
# ═══════════════════════════════════════════════════════════════════

class TestReasoningEngineExtra:
    def test_add_custom_rule(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph
        sg = SceneGraph()
        engine = ReasoningEngine(scene_graph=sg)
        engine.add_custom_rule("path(X,Y) :- edge(X,Y)")

    def test_query_empty(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph
        sg = SceneGraph()
        engine = ReasoningEngine(scene_graph=sg)
        results = engine.query("on")
        assert isinstance(results, list)

    def test_sync_and_query(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
        sg = SceneGraph()
        n1 = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0], attributes={"class": "a"})
        n2 = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 0, 0], attributes={"class": "b"})
        sg.add_edge(n1.id, n2.id, RelationType.ON)
        engine = ReasoningEngine(scene_graph=sg)
        engine.sync_from_scene_graph()
        results = engine.query("on")
        assert isinstance(results, list)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════
# UMA / Memory (83%)
# ═══════════════════════════════════════════════════════════════════

class TestUMAExtra:
    def test_allocate_and_get(self):
        from nesy.core.memory import UMA, DataType, DeviceType
        uma = UMA()
        buf = uma.allocate("test_buf", (10,), DataType.FLOAT32, DeviceType.CPU)
        assert buf is not None
        retrieved = uma.get("test_buf")
        assert retrieved is not None

    def test_allocate_free_reallocate(self):
        from nesy.core.memory import UMA, DataType, DeviceType
        uma = UMA()
        uma.allocate("x", (5,), DataType.FLOAT32, DeviceType.CPU)
        uma.free("x")
        assert uma.get("x") is None
        uma.allocate("x", (10,), DataType.FLOAT32, DeviceType.CPU)
        buf = uma.get("x")
        assert buf is not None

    def test_get_nonexistent(self):
        from nesy.core.memory import UMA
        uma = UMA()
        assert uma.get("nonexistent") is None

    def test_clear_all(self):
        from nesy.core.memory import UMA, DataType, DeviceType
        uma = UMA()
        uma.allocate("a", (5,), DataType.FLOAT32, DeviceType.CPU)
        uma.clear_all()
        assert uma.get("a") is None

    def test_memory_usage(self):
        from nesy.core.memory import UMA
        uma = UMA()
        usage = uma.memory_usage()
        assert isinstance(usage, int)

    def test_list_buffers(self):
        from nesy.core.memory import UMA, DataType, DeviceType
        uma = UMA()
        uma.allocate("buf1", (5,), DataType.FLOAT32, DeviceType.CPU)
        uma.allocate("buf2", (3,), DataType.INT32, DeviceType.CPU)
        buffers = uma.list_buffers()
        assert "buf1" in buffers
        assert "buf2" in buffers
