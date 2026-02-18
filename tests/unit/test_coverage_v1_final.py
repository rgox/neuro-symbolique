"""Comprehensive coverage boost for v1 — targets all modules below 85%."""

import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# NeSyPlatform (__init__.py) — 29% → target 90%+
# ═══════════════════════════════════════════════════════════════════

class TestNeSyPlatform:
    def test_default_init(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p.config is not None
        assert p.uma is not None
        assert p.hal is not None
        assert p.hal.npu is not None
        assert p.hal.spu is not None
        assert p.hal.cpu is not None

    def test_hal_device_pool(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p.hal.device_pool is not None
        # NPU + SPU + CPU registered
        assert len(p.hal.device_pool.devices) >= 3

    def test_perceive_placeholder(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        result = p.perceive("dummy")
        assert result == {}

    def test_reason_placeholder(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        result = p.reason("find cups")
        # reason() now returns [] from the logic engine (no facts loaded)
        assert result == [] or result is None

    def test_visualize_placeholder(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        p.visualize()  # Should not raise

    def test_shutdown(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        p.shutdown()  # Should not raise

    def test_context_manager(self):
        from nesy import NeSyPlatform
        with NeSyPlatform() as p:
            assert p.uma is not None
        # After exit, shutdown called

    def test_context_manager_with_exception(self):
        from nesy import NeSyPlatform
        try:
            with NeSyPlatform() as p:
                raise ValueError("test error")
        except ValueError:
            pass  # shutdown called despite exception

    def test_custom_config(self):
        from nesy import NeSyPlatform
        from nesy.core.config import get_default_config
        config = get_default_config()
        p = NeSyPlatform(config=config)
        assert p.config is config

    def test_from_config_with_file(self, tmp_path):
        from nesy import NeSyPlatform
        config_file = tmp_path / "test.yaml"
        # Minimal YAML — empty means defaults
        config_file.write_text("{}")
        p = NeSyPlatform.from_config(str(config_file))
        assert p.config is not None

    def test_subsystems_none_initially(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p.world_model is None
        # Lazy properties: not initialized until accessed
        assert p._perception_detector is None
        assert p._reasoning_engine is None
        assert p._message_bus is None
        assert p.tools is None


# ═══════════════════════════════════════════════════════════════════
# Tools CLI (tools/main.py) — 0% → target 100%
# ═══════════════════════════════════════════════════════════════════

class TestToolsMain:
    def test_check_command(self):
        from nesy.tools.main import main
        with patch('sys.argv', ['nesy-tools', '--check']):
            main()  # Should not raise

    def test_no_args_prints_help(self, capsys):
        from nesy.tools.main import main
        with patch('sys.argv', ['nesy-tools']):
            main()

    def test_check_with_missing_torch(self):
        from nesy.tools.main import main
        with patch('sys.argv', ['nesy-tools', '--check']):
            with patch.dict('sys.modules', {'torch': None}):
                main()

    def test_check_with_missing_cv2(self):
        from nesy.tools.main import main
        with patch('sys.argv', ['nesy-tools', '--check']):
            with patch.dict('sys.modules', {'cv2': None}):
                main()

    def test_check_with_missing_transformers(self):
        from nesy.tools.main import main
        with patch('sys.argv', ['nesy-tools', '--check']):
            with patch.dict('sys.modules', {'transformers': None}):
                main()

    def test_check_with_missing_ultralytics(self):
        from nesy.tools.main import main
        with patch('sys.argv', ['nesy-tools', '--check']):
            with patch.dict('sys.modules', {'ultralytics': None}):
                main()


# ═══════════════════════════════════════════════════════════════════
# Tools Viz (tools/viz/main.py) — 0% → target 90%+
# ═══════════════════════════════════════════════════════════════════

class TestToolsVizMain:
    def test_scene_graph_viz(self, capsys):
        from nesy.tools.viz.main import main
        with patch('sys.argv', ['nesy-viz', '--scene-graph']):
            main()
        captured = capsys.readouterr()
        assert "Scene Graph" in captured.out or "Visualization" in captured.out

    def test_pipeline_not_implemented(self, capsys):
        from nesy.tools.viz.main import main
        with patch('sys.argv', ['nesy-viz', '--pipeline']):
            main()
        captured = capsys.readouterr()
        assert "not implemented" in captured.out

    def test_no_args_prints_help(self, capsys):
        from nesy.tools.viz.main import main
        with patch('sys.argv', ['nesy-viz']):
            main()


# ═══════════════════════════════════════════════════════════════════
# Detection — bbox utilities + NMS + IoU (object_detection.py)
# ═══════════════════════════════════════════════════════════════════

class TestDetectionDataclass:
    def _make_det(self, **kwargs):
        from nesy.perception.object_detection import Detection
        defaults = dict(
            class_id=0, class_name="person", confidence=0.9,
            bbox=(10, 20, 110, 120), image_size=(640, 480),
        )
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_bbox_center(self):
        d = self._make_det(bbox=(10, 20, 110, 120))
        cx, cy = d.get_bbox_center()
        assert cx == 60.0
        assert cy == 70.0

    def test_bbox_size(self):
        d = self._make_det(bbox=(10, 20, 110, 120))
        w, h = d.get_bbox_size()
        assert w == 100.0
        assert h == 100.0

    def test_bbox_area(self):
        d = self._make_det(bbox=(10, 20, 110, 120))
        assert d.get_bbox_area() == 10000.0

    def test_to_dict(self):
        d = self._make_det()
        result = d.to_dict()
        assert result["class_id"] == 0
        assert result["class_name"] == "person"
        assert result["confidence"] == 0.9
        assert "bbox" in result
        assert "bbox_center" in result
        assert "bbox_size" in result
        assert "bbox_area" in result
        assert "image_size" in result


class TestObjectDetectorNMS:
    def test_compute_iou_overlap(self):
        from nesy.perception.object_detection import ObjectDetector
        iou = ObjectDetector._compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2  # 25/(100+100-25)

    def test_compute_iou_no_overlap(self):
        from nesy.perception.object_detection import ObjectDetector
        iou = ObjectDetector._compute_iou((0, 0, 10, 10), (20, 20, 30, 30))
        assert iou == 0.0

    def test_compute_iou_identical(self):
        from nesy.perception.object_detection import ObjectDetector
        iou = ObjectDetector._compute_iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert iou == 1.0

    def test_apply_nms_single(self):
        from nesy.perception.object_detection import ObjectDetector, Detection
        det = ObjectDetector(backend="mock")
        d = Detection(class_id=0, class_name="a", confidence=0.9,
                      bbox=(0, 0, 10, 10), image_size=(100, 100))
        result = det._apply_nms([d])
        assert len(result) == 1

    def test_apply_nms_overlapping(self):
        from nesy.perception.object_detection import ObjectDetector, Detection
        det = ObjectDetector(backend="mock", nms_threshold=0.3)
        d1 = Detection(class_id=0, class_name="a", confidence=0.9,
                       bbox=(0, 0, 100, 100), image_size=(200, 200))
        d2 = Detection(class_id=0, class_name="a", confidence=0.7,
                       bbox=(10, 10, 110, 110), image_size=(200, 200))
        d3 = Detection(class_id=1, class_name="b", confidence=0.8,
                       bbox=(0, 0, 100, 100), image_size=(200, 200))
        result = det._apply_nms([d1, d2, d3])
        # d2 should be suppressed (same class, high overlap), d3 kept (different class)
        assert len(result) == 2

    def test_apply_nms_empty(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock")
        assert det._apply_nms([]) == []

    def test_class_names_coco(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock")
        assert len(det.class_names) == 80
        assert "person" in det.class_names

    def test_detect_mock_high_threshold(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock", confidence_threshold=0.99)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = det.detect(img)
        assert len(results) == 0  # Both detections below 0.99


# ═══════════════════════════════════════════════════════════════════
# FeatureVector + FeatureExtractor (feature_extraction.py)
# ═══════════════════════════════════════════════════════════════════

class TestFeatureVector:
    def test_normalize(self):
        from nesy.perception.feature_extraction import FeatureVector
        fv = FeatureVector(features=np.array([3.0, 4.0]), dim=2, model="test")
        nfv = fv.normalize()
        assert nfv.normalized
        assert abs(np.linalg.norm(nfv.features) - 1.0) < 1e-5

    def test_normalize_already_normalized(self):
        from nesy.perception.feature_extraction import FeatureVector
        fv = FeatureVector(features=np.array([1.0, 0.0]), dim=2, model="test", normalized=True)
        nfv = fv.normalize()
        assert nfv is fv  # Same object returned

    def test_normalize_zero_vector(self):
        from nesy.perception.feature_extraction import FeatureVector
        fv = FeatureVector(features=np.zeros(3), dim=3, model="test")
        nfv = fv.normalize()
        assert nfv.normalized
        assert np.all(nfv.features == 0)

    def test_cosine_similarity(self):
        from nesy.perception.feature_extraction import FeatureVector
        fv1 = FeatureVector(features=np.array([1.0, 0.0]), dim=2, model="test")
        fv2 = FeatureVector(features=np.array([1.0, 0.0]), dim=2, model="test")
        assert abs(fv1.cosine_similarity(fv2) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        from nesy.perception.feature_extraction import FeatureVector
        fv1 = FeatureVector(features=np.array([1.0, 0.0]), dim=2, model="test")
        fv2 = FeatureVector(features=np.array([0.0, 1.0]), dim=2, model="test")
        assert abs(fv1.cosine_similarity(fv2)) < 1e-5

    def test_mock_no_normalize(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock", normalize=False)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        fv = ext.extract(img)
        assert not fv.normalized

    def test_mock_feature_dim(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        assert ext.feature_dim == 512

    def test_mock_preprocessing_identity(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        data = [1, 2, 3]
        assert ext.preprocess(data) is data


class TestFeatureExtractorEdgeCases:
    def test_invalid_backend(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        with pytest.raises(ValueError):
            FeatureExtractor(backend="nonexistent")

    def test_mock_model_is_none(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        assert ext.model is None

    def test_extract_with_normalize(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock", normalize=True)
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        fv = ext.extract(img)
        assert fv.normalized
        assert abs(np.linalg.norm(fv.features) - 1.0) < 1e-4

    def test_extract_with_crop(self):
        from nesy.perception.feature_extraction import FeatureExtractor
        ext = FeatureExtractor(backend="mock")
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        fv = ext.extract(img, bbox=(10, 10, 50, 50))
        assert fv.dim == 512


class TestObjectDetectorEdgeCases:
    def test_invalid_backend(self):
        from nesy.perception.object_detection import ObjectDetector
        with pytest.raises(ValueError):
            ObjectDetector(backend="nonexistent")

    def test_mock_model_is_none(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock")
        assert det.model is None

    def test_mock_detection_returns_two(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock", confidence_threshold=0.0)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = det.detect(img)
        assert len(results) == 2
        names = {r.class_name for r in results}
        assert names == {"chair", "cup"}

    def test_mock_detection_image_size(self):
        from nesy.perception.object_detection import ObjectDetector
        det = ObjectDetector(backend="mock", confidence_threshold=0.0)
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        results = det.detect(img)
        assert results[0].image_size == (400, 300)


# ═══════════════════════════════════════════════════════════════════
# __main__.py entry points
# ═══════════════════════════════════════════════════════════════════

class TestEntryPoints:
    def test_tools_main_importable(self):
        from nesy.tools.__main__ import main
        assert callable(main)

    def test_viz_main_importable(self):
        from nesy.tools.viz.__main__ import main
        assert callable(main)


# ═══════════════════════════════════════════════════════════════════
# API server coverage boost
# ═══════════════════════════════════════════════════════════════════

class TestAPIServer:
    def test_import_server(self):
        from nesy.api.server import create_app
        app = create_app()
        assert app is not None

    def test_api_init(self):
        import nesy.api
        assert hasattr(nesy.api, '__name__')


# ═══════════════════════════════════════════════════════════════════
# Additional Device coverage
# ═══════════════════════════════════════════════════════════════════

class TestDeviceMisc:
    def test_workload_dataclass(self):
        from nesy.hal.device import Workload, WorkloadType
        wl = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation="test",
            inputs={"a": 1},
        )
        assert wl.workload_type == WorkloadType.GENERAL_COMPUTE
        assert wl.outputs is None

    def test_execution_result_dataclass(self):
        from nesy.hal.device import ExecutionResult, DeviceType
        er = ExecutionResult(
            success=True,
            outputs={"x": 1},
            duration_ms=10.0,
            memory_used_bytes=1024,
            device_type=DeviceType.CPU,
        )
        assert er.success
        assert er.metadata is None or er.metadata == {}

    def test_device_repr(self):
        from nesy.hal.npu.simulator import NPU
        from nesy.core.memory import UMA
        npu = NPU(uma=UMA())
        r = repr(npu)
        assert "NPU" in r or "npu" in r.lower()


# ═══════════════════════════════════════════════════════════════════
# SPU (spu/simulator.py) — 63% → target 90%+
# ═══════════════════════════════════════════════════════════════════

class TestSPUGraph:
    def test_graph_add_node(self):
        from nesy.hal.spu.simulator import Graph
        g = Graph()
        g.add_node("a", {"type": "object"})
        assert "a" in g.nodes
        assert g.node_attributes["a"]["type"] == "object"

    def test_graph_add_edge(self):
        from nesy.hal.spu.simulator import Graph
        g = Graph()
        g.add_edge("a", "knows", "b", {"weight": 1.0})
        assert "a" in g.nodes
        assert "b" in g.nodes
        assert ("knows", "b") in g.edges["a"]
        assert ("a", "knows", "b") in g.edge_attributes

    def test_graph_get_neighbors(self):
        from nesy.hal.spu.simulator import Graph
        g = Graph()
        g.add_edge("a", "r1", "b")
        g.add_edge("a", "r2", "c")
        assert set(g.get_neighbors("a")) == {"b", "c"}
        assert g.get_neighbors("a", "r1") == ["b"]
        assert g.get_neighbors("z") == []

    def test_graph_query(self):
        from nesy.hal.spu.simulator import Graph
        g = Graph()
        g.add_edge("a", "likes", "b")
        g.add_edge("a", "likes", "c")
        g.add_edge("a", "hates", "d")
        results = g.query({"src": "a", "relation": "likes"})
        assert len(results) == 2


class TestSPUExecution:
    def _make_spu(self):
        from nesy.hal.spu.simulator import SPU
        from nesy.core.memory import UMA
        return SPU(uma=UMA())

    def test_bfs_traversal(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        g = spu.get_graph()
        g.add_edge("a", "to", "b")
        g.add_edge("b", "to", "c")
        g.add_edge("c", "to", "d")

        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "a", "goal": "d"},
        )
        result = spu.execute(wl)
        assert result.success
        assert result.outputs["found"]
        assert result.outputs["path"][-1] == "d"

    def test_dfs_traversal(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        g = spu.get_graph()
        g.add_edge("a", "to", "b")
        g.add_edge("b", "to", "c")

        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="dfs",
            inputs={"start": "a", "goal": "c"},
        )
        result = spu.execute(wl)
        assert result.success
        assert result.outputs["found"]

    def test_bfs_no_goal(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        g = spu.get_graph()
        g.add_edge("a", "to", "b")
        g.add_edge("b", "to", "c")

        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "a"},
        )
        result = spu.execute(wl)
        assert result.success
        assert set(result.outputs["path"]) == {"a", "b", "c"}

    def test_bfs_missing_start(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "nonexistent"},
        )
        result = spu.execute(wl)
        assert result.success
        assert result.outputs["path"] == []

    def test_dfs_missing_start(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="dfs",
            inputs={"start": "nonexistent"},
        )
        result = spu.execute(wl)
        assert result.success

    def test_neighbors_operation(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        g = spu.get_graph()
        g.add_edge("a", "to", "b")
        g.add_edge("a", "to", "c")

        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="neighbors",
            inputs={"start": "a"},
        )
        result = spu.execute(wl)
        assert result.success
        assert set(result.outputs["neighbors"]) == {"b", "c"}

    def test_unknown_traversal_op(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="unknown_op",
            inputs={"start": "a"},
        )
        result = spu.execute(wl)
        assert not result.success

    def test_missing_graph(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"graph": "nonexistent", "start": "a"},
        )
        result = spu.execute(wl)
        assert not result.success

    def test_add_fact_and_query(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()

        # Add fact
        wl = Workload(
            workload_type=WorkloadType.LOGIC_EVALUATION,
            operation="add_fact",
            inputs={"fact": ("cup", "on", "table")},
        )
        result = spu.execute(wl)
        assert result.success
        assert result.outputs["total_facts"] == 1

        # Query fact
        wl2 = Workload(
            workload_type=WorkloadType.LOGIC_EVALUATION,
            operation="query_fact",
            inputs={"pattern": ("cup", "on", None)},
        )
        result2 = spu.execute(wl2)
        assert result2.success
        assert result2.outputs["count"] == 1

    def test_query_fact_wildcard(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        spu.facts.add(("a", "r", "b"))
        spu.facts.add(("a", "r", "c"))
        spu.facts.add(("x", "r", "y"))

        wl = Workload(
            workload_type=WorkloadType.LOGIC_EVALUATION,
            operation="query_fact",
            inputs={"pattern": ("a", None, None)},
        )
        result = spu.execute(wl)
        assert result.outputs["count"] == 2

    def test_forward_chain(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.LOGIC_EVALUATION,
            operation="forward_chain",
            inputs={},
        )
        result = spu.execute(wl)
        assert result.success

    def test_unknown_logic_op(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.LOGIC_EVALUATION,
            operation="unknown",
            inputs={},
        )
        result = spu.execute(wl)
        assert not result.success

    def test_symbolic_query(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        g = spu.get_graph()
        g.add_edge("a", "likes", "b")

        wl = Workload(
            workload_type=WorkloadType.SYMBOLIC_QUERY,
            operation="query",
            inputs={"pattern": {"src": "a", "relation": "likes"}},
        )
        result = spu.execute(wl)
        assert result.success
        assert result.outputs["count"] == 1
        assert spu.total_queries == 1

    def test_unsupported_workload(self):
        from nesy.hal.device import Workload, WorkloadType
        spu = self._make_spu()
        wl = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation="model",
            inputs={},
        )
        with pytest.raises(NotImplementedError):
            spu.execute(wl)

    def test_add_graph_and_get(self):
        spu = self._make_spu()
        spu.add_graph("test_graph")
        g = spu.get_graph("test_graph")
        assert g is not None
        assert spu.get_graph("missing") is None

    def test_add_rule(self):
        spu = self._make_spu()
        spu.add_rule({"if": "a", "then": "b"})
        assert len(spu.rules) == 1

    def test_clear_facts(self):
        spu = self._make_spu()
        spu.facts.add(("a", "b", "c"))
        spu.clear_facts()
        assert len(spu.facts) == 0

    def test_clear_graphs(self):
        spu = self._make_spu()
        spu.add_graph("extra")
        spu.clear_graphs()
        assert "default" in spu.graphs
        assert "extra" not in spu.graphs

    def test_estimate_memory(self):
        spu = self._make_spu()
        g = spu.get_graph()
        g.add_edge("a", "to", "b")
        spu.facts.add(("x", "y", "z"))
        mem = spu._estimate_memory_usage()
        assert mem > 0

    def test_capabilities_fields(self):
        spu = self._make_spu()
        caps = spu.get_capabilities()
        assert "mode" in caps
        assert "num_graphs" in caps
        assert "num_facts" in caps
        assert "supported_workloads" in caps


# ═══════════════════════════════════════════════════════════════════
# NPU (npu/simulator.py) — 68% → target 90%+
# ═══════════════════════════════════════════════════════════════════

class TestNPUExecution:
    def _make_npu(self):
        from nesy.hal.npu.simulator import NPU
        from nesy.core.memory import UMA
        return NPU(uma=UMA())

    def test_execute_nn_module(self):
        npu = self._make_npu()
        model = nn.Linear(10, 5)
        from nesy.hal.device import Workload, WorkloadType
        wl = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=model,
            inputs={"input": torch.randn(1, 10)},
        )
        result = npu.execute(wl)
        assert result.success
        assert "outputs" in result.outputs

    def test_execute_callable(self):
        npu = self._make_npu()
        from nesy.hal.device import Workload, WorkloadType
        wl = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=lambda x: x * 2,
            inputs={"input": torch.tensor([1.0, 2.0, 3.0])},
        )
        result = npu.execute(wl)
        assert result.success

    def test_execute_invalid_op(self):
        npu = self._make_npu()
        from nesy.hal.device import Workload, WorkloadType
        wl = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation="not_callable",
            inputs={"input": torch.randn(2)},
        )
        result = npu.execute(wl)
        assert not result.success

    def test_execute_unsupported_workload(self):
        npu = self._make_npu()
        from nesy.hal.device import Workload, WorkloadType
        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={},
        )
        with pytest.raises(NotImplementedError):
            npu.execute(wl)

    def test_prepare_inputs_tensor(self):
        npu = self._make_npu()
        t = torch.randn(3)
        result = npu._prepare_inputs({"input": t})
        assert isinstance(result, torch.Tensor)

    def test_prepare_inputs_numpy(self):
        npu = self._make_npu()
        arr = np.array([1.0, 2.0, 3.0])
        result = npu._prepare_inputs({"input": arr})
        assert isinstance(result, torch.Tensor)

    def test_prepare_inputs_list(self):
        npu = self._make_npu()
        result = npu._prepare_inputs({"input": [1.0, 2.0]})
        assert isinstance(result, torch.Tensor)

    def test_prepare_inputs_tensor_key(self):
        npu = self._make_npu()
        result = npu._prepare_inputs({"tensor": torch.randn(3)})
        assert isinstance(result, torch.Tensor)

    def test_prepare_inputs_first_value(self):
        npu = self._make_npu()
        result = npu._prepare_inputs({"custom_key": torch.randn(3)})
        assert isinstance(result, torch.Tensor)

    def test_prepare_inputs_buffer_key_missing(self):
        npu = self._make_npu()
        with pytest.raises(ValueError, match="not found"):
            npu._prepare_inputs({"input": "nonexistent_buffer"})

    def test_store_outputs_auto_keys(self):
        npu = self._make_npu()
        t = torch.randn(5)
        keys = npu._store_outputs(t, None)
        assert len(keys) == 1
        assert keys[0].startswith("npu_output_")

    def test_store_outputs_explicit_keys(self):
        npu = self._make_npu()
        t = torch.randn(5)
        keys = npu._store_outputs(t, ["my_output"])
        assert keys == ["my_output"]

    def test_store_outputs_multiple(self):
        npu = self._make_npu()
        t1 = torch.randn(5)
        t2 = torch.randn(3)
        keys = npu._store_outputs([t1, t2], ["out1", "out2"])
        assert len(keys) == 2

    def test_store_outputs_dtypes(self):
        npu = self._make_npu()
        for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
            t = torch.zeros(3, dtype=dtype)
            keys = npu._store_outputs(t, [f"test_{dtype}"])
            assert len(keys) == 1

    def test_get_memory_usage(self):
        npu = self._make_npu()
        mem = npu._get_memory_usage()
        assert isinstance(mem, int)

    def test_execute_model_convenience(self):
        npu = self._make_npu()
        model = nn.Linear(10, 5)
        result = npu.execute_model(model, torch.randn(1, 10))
        assert result.success

    def test_execute_model_with_key(self):
        npu = self._make_npu()
        model = nn.Linear(10, 5)
        result = npu.execute_model(model, torch.randn(1, 10), output_key="my_out")
        assert result.success

    def test_profile_model(self):
        npu = self._make_npu()
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        profile = npu.profile_model(model, (1, 10))
        assert profile["total_params"] > 0
        assert profile["trainable_params"] > 0
        assert profile["forward_time_ms"] >= 0
        assert profile["estimated_flops"] > 0  # Has Linear layers
        assert "memory_params_mb" in profile

    def test_supports_general_compute(self):
        npu = self._make_npu()
        from nesy.hal.device import WorkloadType
        assert npu.supports_workload(WorkloadType.GENERAL_COMPUTE)
        assert npu.supports_workload(WorkloadType.NEURAL_TRAINING)

    def test_capabilities_fields(self):
        npu = self._make_npu()
        caps = npu.get_capabilities()
        assert "backend" in caps
        assert "device" in caps
        assert "precision" in caps
        assert "cuda_available" in caps


# ═══════════════════════════════════════════════════════════════════
# CPU Orchestrator — 59% → target 90%+
# ═══════════════════════════════════════════════════════════════════

class TestCPUOrchestratorAdvanced:
    def _make_cpu(self):
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.device import DevicePool
        from nesy.core.memory import UMA
        uma = UMA()
        return CPUOrchestrator(uma=uma, device_pool=DevicePool(uma=uma))

    def test_execute_non_callable(self):
        from nesy.hal.device import Workload, WorkloadType
        cpu = self._make_cpu()
        wl = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation="not_callable",
            inputs={"x": 5},
        )
        result = cpu.execute(wl)
        assert result.success
        assert result.outputs["result"] == "CPU fallback execution"

    def test_execute_failure(self):
        from nesy.hal.device import Workload, WorkloadType
        cpu = self._make_cpu()

        def bad_op():
            raise RuntimeError("boom")

        wl = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation=bad_op,
            inputs={},
        )
        result = cpu.execute(wl)
        assert not result.success
        assert "boom" in result.metadata.get("error", "")

    def test_execute_pipeline(self):
        from nesy.hal.device import Workload, WorkloadType
        cpu = self._make_cpu()
        workloads = [
            Workload(WorkloadType.GENERAL_COMPUTE, lambda: {"v": 1}, {}),
            Workload(WorkloadType.GENERAL_COMPUTE, lambda: {"v": 2}, {}),
        ]
        results = cpu.execute_pipeline(workloads)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_execute_pipeline_failure_stops(self):
        from nesy.hal.device import Workload, WorkloadType
        cpu = self._make_cpu()

        def fail():
            raise RuntimeError("fail")

        workloads = [
            Workload(WorkloadType.GENERAL_COMPUTE, lambda: {"v": 1}, {}),
            Workload(WorkloadType.GENERAL_COMPUTE, fail, {}),
            Workload(WorkloadType.GENERAL_COMPUTE, lambda: {"v": 3}, {}),
        ]
        results = cpu.execute_pipeline(workloads)
        assert len(results) == 2  # Stopped after failure
        assert results[0].success
        assert not results[1].success

    def test_execute_parallel(self):
        from nesy.hal.device import Workload, WorkloadType
        cpu = self._make_cpu()
        workloads = [
            Workload(WorkloadType.GENERAL_COMPUTE, lambda: {"v": i}, {})
            for i in range(3)
        ]
        results = cpu.execute_parallel(workloads)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_get_device_statistics(self):
        cpu = self._make_cpu()
        stats = cpu.get_device_statistics()
        assert isinstance(stats, dict)

    def test_shutdown(self):
        cpu = self._make_cpu()
        cpu.shutdown()  # Should not raise

    def test_with_device_routing(self):
        """Test that workloads are routed to specialized devices."""
        from nesy.hal.cpu.orchestrator import CPUOrchestrator
        from nesy.hal.spu.simulator import SPU
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.core.memory import UMA

        uma = UMA()
        pool = DevicePool(uma=uma)
        spu = SPU(uma=uma)
        spu.get_graph().add_edge("a", "to", "b")
        pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=pool)

        wl = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "a"},
        )
        result = cpu.execute(wl)
        assert result.success
        # Routed to SPU, not CPU fallback
        assert result.device_type.value == "spu"


# ═══════════════════════════════════════════════════════════════════
# DevicePool (device.py) — 80% → target 95%+
# ═══════════════════════════════════════════════════════════════════

class TestDevicePoolAdvanced:
    def test_register_and_find(self):
        from nesy.hal.device import DevicePool, Workload, WorkloadType
        from nesy.hal.spu.simulator import SPU
        from nesy.core.memory import UMA
        uma = UMA()
        pool = DevicePool(uma=uma)
        spu = SPU(uma=uma)
        pool.register_device(spu)

        wl = Workload(WorkloadType.GRAPH_TRAVERSAL, "bfs", {})
        device = pool.find_device_for_workload(wl)
        assert device is spu

    def test_get_all_statistics(self):
        from nesy.hal.device import DevicePool
        from nesy.core.memory import UMA
        pool = DevicePool(uma=UMA())
        stats = pool.get_all_statistics()
        assert isinstance(stats, dict)

    def test_repr(self):
        from nesy.hal.device import DevicePool
        from nesy.core.memory import UMA
        pool = DevicePool(uma=UMA())
        r = repr(pool) if hasattr(pool, '__repr__') else str(pool)
        assert r is not None
