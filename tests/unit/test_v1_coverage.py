"""Coverage boost tests for v1 — targeting modules below 85%."""

import pytest
import numpy as np


# ── ScallopContext (Minalog fallback) ──────────────────────────────

class TestScallopContextMinalog:
    def test_basic_lifecycle(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext, SCALLOP_AVAILABLE
        ctx = ScallopContext()
        ctx.add_relation("edge", ["String", "String"])
        ctx.add_fact("edge", "a", "b")
        ctx.add_fact("edge", "b", "c")
        ctx.add_rule("path(X, Y) :- edge(X, Y)")
        ctx.add_rule("path(X, Z) :- edge(X, Y), path(Y, Z)")
        results = ctx.query("path")
        assert ("a", "b") in results
        assert ("a", "c") in results

    def test_add_facts_batch(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        ctx.add_relation("r", ["String", "String"])
        ctx.add_facts_batch("r", [("a", "b"), ("c", "d")])
        assert ctx.get_num_facts() == 2

    def test_add_rules_batch(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        ctx.add_relation("a", ["String"])
        ctx.add_relation("b", ["String"])
        ctx.add_rules_batch(["b(X) :- a(X)"])
        assert ctx.get_num_rules() == 1

    def test_query_with_provenance(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        ctx.add_relation("r", ["String"])
        ctx.add_fact("r", "x")
        results = ctx.query_with_provenance("r")
        assert len(results) >= 1
        assert results[0][1] == 1.0  # simplified provenance

    def test_clear_facts(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        ctx.add_relation("r", ["String"])
        ctx.add_fact("r", "x")
        ctx.add_rule("s(X) :- r(X)")
        ctx.clear_facts()
        assert ctx.get_num_facts() == 0
        assert ctx.get_num_rules() == 1  # rules preserved

    def test_clear_all(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        ctx.add_relation("r", ["String"])
        ctx.add_fact("r", "x")
        ctx.add_rule("s(X) :- r(X)")
        ctx.clear_all()
        assert ctx.get_num_facts() == 0
        assert ctx.get_num_rules() == 0
        assert ctx.get_relations() == []

    def test_get_relations(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        ctx.add_relation("edge", ["String", "String"])
        ctx.add_relation("node", ["String"])
        rels = ctx.get_relations()
        assert "edge" in rels
        assert "node" in rels

    def test_repr(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        r = repr(ctx)
        assert "ScallopContext" in r
        assert "facts=0" in r

    def test_scallop_available_flag(self):
        from nesy.reasoning.logic.scallop_context import SCALLOP_AVAILABLE
        assert isinstance(SCALLOP_AVAILABLE, bool)

    def test_create_context_helper(self):
        from nesy.reasoning.logic.scallop_context import ScallopContext
        ctx = ScallopContext()
        new_ctx = ctx._create_context()
        assert new_ctx is not None


# ── Advanced Reasoning ──────────────────────────────────────────────

class TestAdvancedReasoningBFS:
    def test_find_path_bfs(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
        sg = SceneGraph()
        n1 = sg.add_node(layer=LayerType.L1, node_type=NodeType.OBJECT,
                         position=[0, 0, 0], attributes={"class": "a"})
        n2 = sg.add_node(layer=LayerType.L1, node_type=NodeType.OBJECT,
                         position=[1, 0, 0], attributes={"class": "b"})
        n3 = sg.add_node(layer=LayerType.L1, node_type=NodeType.OBJECT,
                         position=[2, 0, 0], attributes={"class": "c"})
        sg.add_edge(n1.id, n2.id, RelationType.NEAR)
        sg.add_edge(n2.id, n3.id, RelationType.NEAR)
        engine = ReasoningEngine(scene_graph=sg)
        engine.sync_from_scene_graph()
        from nesy.reasoning.logic.advanced import RecursiveRules
        rr = RecursiveRules(engine)
        path = rr.find_path(n1.id, n3.id)
        assert path is not None
        assert path[0] == n1.id
        assert path[-1] == n3.id

    def test_find_path_no_path(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        sg = SceneGraph()
        n1 = sg.add_node(layer=LayerType.L1, node_type=NodeType.OBJECT,
                         position=[0, 0, 0], attributes={"class": "a"})
        n2 = sg.add_node(layer=LayerType.L1, node_type=NodeType.OBJECT,
                         position=[10, 10, 10], attributes={"class": "b"})
        # No edge between them
        engine = ReasoningEngine(scene_graph=sg)
        engine.sync_from_scene_graph()
        from nesy.reasoning.logic.advanced import RecursiveRules
        rr = RecursiveRules(engine)
        path = rr.find_path(n1.id, n2.id)
        assert path is None

    def test_find_path_nonexistent_node(self):
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph
        sg = SceneGraph()
        engine = ReasoningEngine(scene_graph=sg)
        from nesy.reasoning.logic.advanced import RecursiveRules
        rr = RecursiveRules(engine)
        path = rr.find_path("fake_id_1", "fake_id_2")
        assert path is None


# ── HAL Coverage Boost ──────────────────────────────────────────────

class TestHALDevicePool:
    def test_device_pool_find_best(self):
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.core.memory import UMA
        uma = UMA()
        pool = DevicePool(uma=uma)
        # Empty pool
        wl = Workload(workload_type=WorkloadType.NEURAL_INFERENCE, operation="test", inputs={})
        result = pool.find_device_for_workload(wl)
        assert result is None

    def test_device_pool_list_devices(self):
        from nesy.hal.device import DevicePool
        from nesy.core.memory import UMA
        pool = DevicePool(uma=UMA())
        assert len(pool.devices) == 0


class TestNPU:
    def test_npu_supports_workload(self):
        from nesy.hal.npu.simulator import NPU
        from nesy.hal.device import WorkloadType
        from nesy.core.memory import UMA
        npu = NPU(uma=UMA())
        assert npu.supports_workload(WorkloadType.NEURAL_INFERENCE)
        assert not npu.supports_workload(WorkloadType.LOGIC_EVALUATION)

    def test_npu_capabilities(self):
        from nesy.hal.npu.simulator import NPU
        from nesy.core.memory import UMA
        npu = NPU(uma=UMA())
        caps = npu.get_capabilities()
        assert "precision" in caps


class TestSPU:
    def test_spu_supports_workload(self):
        from nesy.hal.spu.simulator import SPU
        from nesy.hal.device import WorkloadType
        from nesy.core.memory import UMA
        spu = SPU(uma=UMA())
        assert spu.supports_workload(WorkloadType.GRAPH_TRAVERSAL)
        assert spu.supports_workload(WorkloadType.LOGIC_EVALUATION)
        assert not spu.supports_workload(WorkloadType.NEURAL_INFERENCE)

    def test_spu_capabilities(self):
        from nesy.hal.spu.simulator import SPU
        from nesy.core.memory import UMA
        spu = SPU(uma=UMA())
        caps = spu.get_capabilities()
        assert isinstance(caps, dict)


class TestCPUOrchestrator:
    def test_cpu_supports_workload(self):
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.device import DevicePool, WorkloadType
        from nesy.core.memory import UMA
        uma = UMA()
        cpu = CPUOrchestrator(uma=uma, device_pool=DevicePool(uma=uma))
        assert cpu.supports_workload(WorkloadType.GENERAL_COMPUTE)

    def test_cpu_capabilities(self):
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.device import DevicePool
        from nesy.core.memory import UMA
        uma = UMA()
        cpu = CPUOrchestrator(uma=uma, device_pool=DevicePool(uma=uma))
        caps = cpu.get_capabilities()
        assert isinstance(caps, dict)

    def test_cpu_execute_callable(self):
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.core.memory import UMA
        uma = UMA()
        cpu = CPUOrchestrator(uma=uma, device_pool=DevicePool(uma=uma))
        wl = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation=lambda x: {"result": x * 2},
            inputs={"x": 5}
        )
        result = cpu.execute(wl)
        assert result.success
        assert result.outputs["result"] == 10

    def test_cpu_execute_dict_result(self):
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.core.memory import UMA
        uma = UMA()
        cpu = CPUOrchestrator(uma=uma, device_pool=DevicePool(uma=uma))
        wl = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation=lambda: {"output": 42},
            inputs={}
        )
        result = cpu.execute(wl)
        assert result.success

    def test_cpu_execute_scalar_result(self):
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.core.memory import UMA
        uma = UMA()
        cpu = CPUOrchestrator(uma=uma, device_pool=DevicePool(uma=uma))
        wl = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation=lambda x: x * 3,
            inputs={"x": 7}
        )
        result = cpu.execute(wl)
        assert result.success
        assert result.outputs["result"] == 21


# ── NLQuery boost ───────────────────────────────────────────────────

class TestNLQueryPatterns:
    def test_where_is_pattern(self):
        from nesy.reasoning.nlquery import NLQueryInterface
        from nesy.reasoning.logic.engine import ReasoningEngine
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
        sg = SceneGraph()
        cup = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1], attributes={"class": "cup"})
        table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 0], attributes={"class": "table"})
        sg.add_edge(cup.id, table.id, RelationType.ON)
        engine = ReasoningEngine(scene_graph=sg)
        engine.sync_from_scene_graph()
        nli = NLQueryInterface(engine, sg)
        result = nli.query("where is cup")
        assert result is not None


# ── Perception mock backends ────────────────────────────────────────

class TestMockDetector:
    def test_mock_returns_detections(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = det.detect(image)
        assert len(results) >= 1
        assert results[0].class_name in ("chair", "cup")

    def test_mock_confidence_filter(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock", confidence_threshold=0.9)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = det.detect(image)
        for r in results:
            assert r.confidence >= 0.9

    def test_mock_class_filter(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock", class_filter=["cup"])
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = det.detect(image)
        for r in results:
            assert r.class_name == "cup"


class TestMockExtractor:
    def test_mock_returns_features(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        fv = ext.extract(image)
        assert fv.dim == 512
        assert fv.features.shape == (512,)

    def test_mock_with_bbox(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        fv = ext.extract(image, bbox=(10, 10, 100, 100))
        assert fv.dim == 512

    def test_mock_batch(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        results = ext.extract_batch(images)
        assert len(results) == 3
