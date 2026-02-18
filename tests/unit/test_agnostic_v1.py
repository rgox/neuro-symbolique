"""
Tests for the v1 Backend-Agnostic Platform Infrastructure.

Covers:
- Registry + register_plugin decorator
- ComponentFactory
- Perception ABCs + plugin backends (mock, yolo, faster_rcnn, clip, resnet, dinov2)
- Reasoning ABCs + plugin backends (scallop, minalog, pddl)
- LLM ABCs + plugin backends (mock, anthropic, openai)
- Agent ABCs + plugin backends (autonomous)
- Middleware plugin registration
- NeSyPlatform lazy DI properties (detector, extractor, logic_engine, llm, message_bus)
- Platform perceive() + reason() with real registry-backed components
- COCO_NAMES constant
- get_registry_summary()
"""

import pytest
import numpy as np
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════
# Registry & Factory
# ═══════════════════════════════════════════════════════════════════

class TestRegistry:
    def setup_method(self):
        from nesy.core.registry import Registry
        self._backup = dict(Registry._registries)

    def teardown_method(self):
        from nesy.core.registry import Registry
        Registry._registries = self._backup

    def test_register_and_create(self):
        from nesy.core.registry import Registry
        from abc import ABC, abstractmethod

        class IFoo(ABC):
            @abstractmethod
            def do(self): ...

        class ConcreteFoo(IFoo):
            def __init__(self, val=1):
                self.val = val
            def do(self):
                return self.val

        Registry.register(IFoo, "concrete", ConcreteFoo)
        instance = Registry.create(IFoo, "concrete", val=42)
        assert instance.do() == 42

    def test_list_plugins(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IBar(ABC): pass
        class Bar1(IBar): pass
        class Bar2(IBar): pass

        Registry.register(IBar, "b1", Bar1)
        Registry.register(IBar, "b2", Bar2)
        plugins = Registry.list_plugins(IBar)
        assert set(plugins) == {"b1", "b2"}

    def test_has_plugin(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IX(ABC): pass
        class X(IX): pass
        Registry.register(IX, "x", X)
        assert Registry.has_plugin(IX, "x")
        assert not Registry.has_plugin(IX, "nonexistent")

    def test_get_class(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IY(ABC): pass
        class Y(IY): pass
        Registry.register(IY, "y", Y)
        assert Registry.get_class(IY, "y") is Y

    def test_get_class_not_found_raises(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IZ(ABC): pass
        with pytest.raises(KeyError, match="No plugins registered"):
            Registry.get_class(IZ, "z")

    def test_get_class_plugin_not_found_raises(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IW(ABC): pass
        class W(IW): pass
        Registry.register(IW, "w", W)
        with pytest.raises(KeyError, match="not found"):
            Registry.get_class(IW, "missing")

    def test_clear_specific_interface(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IA(ABC): pass
        class A(IA): pass
        Registry.register(IA, "a", A)
        Registry.clear(IA)
        assert Registry.list_plugins(IA) == []

    def test_clear_all(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IB(ABC): pass
        class B(IB): pass
        Registry.register(IB, "b", B)
        Registry.clear()
        assert Registry.get_all_interfaces() == []

    def test_get_all_interfaces(self):
        from nesy.core.registry import Registry
        from abc import ABC

        class IC(ABC): pass
        class C(IC): pass
        Registry.register(IC, "c", C)
        assert IC in Registry.get_all_interfaces()


class TestRegisterPluginDecorator:
    def test_decorator_registers_class(self):
        from nesy.core.registry import Registry, register_plugin
        from abc import ABC, abstractmethod

        class IService(ABC):
            @abstractmethod
            def serve(self): ...

        @register_plugin(IService, "test_svc")
        class TestService(IService):
            def serve(self):
                return "served"

        instance = Registry.create(IService, "test_svc")
        assert instance.serve() == "served"
        # Cleanup
        Registry.clear(IService)


class TestComponentFactory:
    def setup_method(self):
        from nesy.core.registry import Registry
        self._backup = dict(Registry._registries)

    def teardown_method(self):
        from nesy.core.registry import Registry
        Registry._registries = self._backup

    def test_create_from_config(self):
        from nesy.core.registry import Registry
        from nesy.core.factory import ComponentFactory
        from abc import ABC

        class IDetector(ABC): pass
        class MockDet(IDetector):
            def __init__(self, confidence=0.5):
                self.confidence = confidence

        Registry.register(IDetector, "mock", MockDet)
        det = ComponentFactory.create(IDetector, {"backend": "mock", "confidence": 0.8})
        assert det.confidence == 0.8

    def test_create_missing_backend_key_raises(self):
        from nesy.core.factory import ComponentFactory
        from abc import ABC

        class IFoo(ABC): pass
        with pytest.raises(KeyError, match="backend"):
            ComponentFactory.create(IFoo, {"not_backend": "mock"})

    def test_create_with_default(self):
        from nesy.core.registry import Registry
        from nesy.core.factory import ComponentFactory
        from abc import ABC

        class IExt(ABC): pass
        class MockExt(IExt):
            def __init__(self, dim=512):
                self.dim = dim

        Registry.register(IExt, "mock", MockExt)
        ext = ComponentFactory.create_with_default(IExt, default_backend="mock")
        assert ext.dim == 512

    def test_create_with_default_none_config(self):
        from nesy.core.registry import Registry
        from nesy.core.factory import ComponentFactory
        from abc import ABC

        class IExt2(ABC): pass
        class MockExt2(IExt2):
            def __init__(self): pass

        Registry.register(IExt2, "mock", MockExt2)
        ext = ComponentFactory.create_with_default(IExt2, config=None, default_backend="mock")
        assert isinstance(ext, MockExt2)


# ═══════════════════════════════════════════════════════════════════
# Perception ABCs + Plugins
# ═══════════════════════════════════════════════════════════════════

class TestPerceptionPlugins:
    def test_detector_plugins_registered(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import ObjectDetectorBase
        import nesy.perception.detectors  # noqa
        plugins = Registry.list_plugins(ObjectDetectorBase)
        assert "mock" in plugins
        assert "yolo" in plugins
        assert "faster_rcnn" in plugins

    def test_extractor_plugins_registered(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import FeatureExtractorBase
        import nesy.perception.extractors  # noqa
        plugins = Registry.list_plugins(FeatureExtractorBase)
        assert "mock" in plugins
        assert "clip" in plugins
        assert "resnet" in plugins
        assert "dinov2" in plugins

    def test_mock_detector_via_registry(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import ObjectDetectorBase
        import nesy.perception.detectors  # noqa

        det = Registry.create(ObjectDetectorBase, "mock", confidence_threshold=0.5)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = det.detect(img)
        assert len(results) == 2
        assert results[0].class_name == "chair"
        assert results[1].class_name == "cup"

    def test_mock_detector_class_filter(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import ObjectDetectorBase
        import nesy.perception.detectors  # noqa

        det = Registry.create(ObjectDetectorBase, "mock", class_filter=["cup"])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = det.detect(img)
        assert len(results) == 1
        assert results[0].class_name == "cup"

    def test_mock_detector_get_class_names(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import ObjectDetectorBase
        import nesy.perception.detectors  # noqa

        det = Registry.create(ObjectDetectorBase, "mock")
        names = det.get_class_names()
        assert "person" in names
        assert "cup" in names
        assert len(names) == 80

    def test_mock_extractor_via_registry(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import FeatureExtractorBase
        import nesy.perception.extractors  # noqa

        ext = Registry.create(FeatureExtractorBase, "mock", feature_dim=256)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        fv = ext.extract(img)
        assert fv.dim == 256
        assert fv.normalized is True

    def test_mock_extractor_get_feature_dim(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import FeatureExtractorBase
        import nesy.perception.extractors  # noqa

        ext = Registry.create(FeatureExtractorBase, "mock", feature_dim=1024)
        assert ext.get_feature_dim() == 1024

    def test_mock_extractor_batch(self):
        from nesy.core.registry import Registry
        from nesy.perception.base import FeatureExtractorBase
        import nesy.perception.extractors  # noqa

        ext = Registry.create(FeatureExtractorBase, "mock")
        imgs = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        results = ext.extract_batch(imgs)
        assert len(results) == 3
        for fv in results:
            assert fv.dim == 512

    def test_coco_names_constant(self):
        from nesy.perception.object_detection import COCO_NAMES
        assert len(COCO_NAMES) == 80
        assert COCO_NAMES[0] == "person"
        assert "cup" in COCO_NAMES

    def test_yolo_detector_lazy_load(self):
        """YOLO detector should NOT load model at init (lazy)."""
        from nesy.perception.detectors.yolo import YOLODetector
        det = YOLODetector(model="yolov8n")
        assert det._model is None  # Not loaded yet

    def test_faster_rcnn_detector_lazy_load(self):
        from nesy.perception.detectors.faster_rcnn import FasterRCNNDetector
        det = FasterRCNNDetector()
        assert det._model is None

    def test_clip_extractor_lazy_load(self):
        from nesy.perception.extractors.clip import CLIPExtractor
        ext = CLIPExtractor()
        assert ext._model is None

    def test_resnet_extractor_lazy_load(self):
        from nesy.perception.extractors.resnet import ResNetExtractor
        ext = ResNetExtractor()
        assert ext._model is None

    def test_dinov2_extractor_lazy_load(self):
        from nesy.perception.extractors.dinov2 import DINOv2Extractor
        ext = DINOv2Extractor()
        assert ext._model is None

    def test_dinov2_feature_dims(self):
        from nesy.perception.extractors.dinov2 import DINOv2Extractor
        assert DINOv2Extractor(model="dinov2_vits14").get_feature_dim() == 384
        assert DINOv2Extractor(model="dinov2_vitb14").get_feature_dim() == 768
        assert DINOv2Extractor(model="dinov2_vitl14").get_feature_dim() == 1024
        assert DINOv2Extractor(model="dinov2_vitg14").get_feature_dim() == 1536

    def test_clip_feature_dims(self):
        from nesy.perception.extractors.clip import CLIPExtractor
        assert CLIPExtractor(model="ViT-B/32").get_feature_dim() == 512
        assert CLIPExtractor(model="ViT-L/14").get_feature_dim() == 768
        assert CLIPExtractor(model="RN50").get_feature_dim() == 1024

    def test_resnet_feature_dim(self):
        from nesy.perception.extractors.resnet import ResNetExtractor
        assert ResNetExtractor().get_feature_dim() == 2048


# ═══════════════════════════════════════════════════════════════════
# Reasoning ABCs + Plugins
# ═══════════════════════════════════════════════════════════════════

class TestReasoningPlugins:
    def test_logic_plugins_registered(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import LogicEngineBase
        import nesy.reasoning.engines  # noqa
        plugins = Registry.list_plugins(LogicEngineBase)
        assert "scallop" in plugins
        assert "minalog" in plugins

    def test_planner_plugins_registered(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import PlannerBase
        import nesy.reasoning.engines  # noqa
        assert "pddl" in Registry.list_plugins(PlannerBase)

    def test_minalog_engine_full_cycle(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import LogicEngineBase
        import nesy.reasoning.engines  # noqa

        engine = Registry.create(LogicEngineBase, "minalog")
        engine.add_relation("edge", ["String", "String"])
        engine.add_fact("edge", "a", "b")
        engine.add_fact("edge", "b", "c")
        engine.add_rule("path(X, Z) :- edge(X, Z)")
        engine.add_rule("path(X, Z) :- edge(X, Y), path(Y, Z)")

        results = engine.query("path")
        # Should find: (a,b), (b,c), (a,c)
        result_set = set(results)
        assert ("a", "b") in result_set
        assert ("b", "c") in result_set
        assert ("a", "c") in result_set

    def test_minalog_engine_stats(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import LogicEngineBase
        import nesy.reasoning.engines  # noqa

        engine = Registry.create(LogicEngineBase, "minalog")
        engine.add_relation("on", ["String", "String"])
        engine.add_fact("on", "cup", "table")
        engine.add_rule("contains(Y, X) :- on(X, Y)")

        assert engine.get_num_facts() >= 1
        assert engine.get_num_rules() >= 1
        assert isinstance(engine.get_relations(), list)

    def test_minalog_engine_clear_facts(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import LogicEngineBase
        import nesy.reasoning.engines  # noqa

        engine = Registry.create(LogicEngineBase, "minalog")
        engine.add_relation("on", ["String", "String"])
        engine.add_fact("on", "cup", "table")
        assert engine.get_num_facts() >= 1
        engine.clear_facts()
        assert engine.get_num_facts() == 0

    def test_scallop_engine_via_registry(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import LogicEngineBase
        import nesy.reasoning.engines  # noqa

        engine = Registry.create(LogicEngineBase, "scallop")
        engine.add_relation("on", ["String", "String"])
        engine.add_fact("on", "cup", "table")
        results = engine.query("on")
        assert ("cup", "table") in results

    def test_pddl_planner_via_registry(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.base import PlannerBase
        import nesy.reasoning.engines  # noqa

        planner = Registry.create(PlannerBase, "pddl")
        assert planner.get_available_actions() == []


# ═══════════════════════════════════════════════════════════════════
# LLM ABCs + Plugins
# ═══════════════════════════════════════════════════════════════════

class TestLLMPlugins:
    def test_llm_plugins_registered(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa
        plugins = Registry.list_plugins(LLMProviderBase)
        assert "mock" in plugins
        assert "anthropic" in plugins
        assert "openai" in plugins

    def test_mock_llm_complete(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        response = llm.complete("Extract rules from this scene")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_llm_complete_rule_prompt(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        response = llm.complete("Generate Datalog rules for spatial reasoning")
        assert "rule" in response.lower() or ":-" in response

    def test_mock_llm_complete_explain(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        response = llm.complete("Explain why cup is in kitchen")
        assert "reasoning" in response.lower() or "transitivity" in response.lower()

    def test_mock_llm_complete_validate(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        response = llm.complete("Please validate these rules for correctness")
        assert isinstance(response, str) and len(response) > 0

    def test_mock_llm_extract_rules(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        rules = llm.extract_rules("cups are on tables, tables are in the kitchen")
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert "rule" in rules[0]

    def test_mock_llm_extract_rules_with_datalog(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        # Input containing Datalog-like rules should be extracted
        rules = llm.extract_rules("in(X, Z) :- on(X, Y), in(Y, Z)")
        assert any(":-" in r["rule"] for r in rules)

    def test_mock_llm_model_info(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        info = llm.get_model_info()
        assert info["provider"] == "mock"
        assert "model" in info

    def test_mock_llm_call_count(self):
        from nesy.core.registry import Registry
        from nesy.reasoning.llm.base import LLMProviderBase
        import nesy.reasoning.llm.providers  # noqa

        llm = Registry.create(LLMProviderBase, "mock")
        assert llm.call_count == 0
        llm.complete("test")
        assert llm.call_count == 1
        llm.extract_rules("test")
        assert llm.call_count == 2

    def test_anthropic_provider_lazy(self):
        from nesy.reasoning.llm.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")
        assert provider._client is None  # Not initialized yet
        assert provider.model == "claude-sonnet-4-5-20250929"
        info = provider.get_model_info()
        assert info["provider"] == "anthropic"

    def test_openai_provider_lazy(self):
        from nesy.reasoning.llm.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(model="gpt-4o")
        assert provider._client is None
        assert provider.model == "gpt-4o"
        info = provider.get_model_info()
        assert info["provider"] == "openai"


# ═══════════════════════════════════════════════════════════════════
# Agents ABC + Plugin
# ═══════════════════════════════════════════════════════════════════

class TestAgentPlugins:
    def test_agent_plugins_registered(self):
        from nesy.core.registry import Registry
        from nesy.agents.base import AgentBase
        import nesy.agents.plugins  # noqa
        assert "autonomous" in Registry.list_plugins(AgentBase)

    def test_autonomous_agent_via_registry(self):
        from nesy.core.registry import Registry
        from nesy.agents.base import AgentBase
        import nesy.agents.plugins  # noqa

        agent = Registry.create(AgentBase, "autonomous")
        state = agent.get_state()
        assert "state" in state
        assert state["state"] == "idle"

    def test_autonomous_agent_observe_non_image(self):
        from nesy.core.registry import Registry
        from nesy.agents.base import AgentBase
        import nesy.agents.plugins  # noqa

        agent = Registry.create(AgentBase, "autonomous")
        result = agent.observe("not an image")
        assert result == {"status": "no_observation"}

    def test_autonomous_agent_reason(self):
        from nesy.core.registry import Registry
        from nesy.agents.base import AgentBase
        import nesy.agents.plugins  # noqa

        agent = Registry.create(AgentBase, "autonomous", config={
            "use_yolo": False, "use_clip": False,
        })
        result = agent.reason()
        assert isinstance(result, dict)

    def test_autonomous_agent_plan_no_goal(self):
        from nesy.core.registry import Registry
        from nesy.agents.base import AgentBase
        import nesy.agents.plugins  # noqa

        agent = Registry.create(AgentBase, "autonomous", config={
            "use_yolo": False, "use_clip": False,
        })
        plan = agent.plan({"description": "find cup"})
        assert isinstance(plan, list)


# ═══════════════════════════════════════════════════════════════════
# Middleware Plugin
# ═══════════════════════════════════════════════════════════════════

class TestMiddlewarePlugin:
    def test_message_bus_registered(self):
        from nesy.core.registry import Registry
        from nesy.middleware.bus import MessageBus
        assert "local" in Registry.list_plugins(MessageBus)

    def test_message_bus_via_registry(self):
        from nesy.core.registry import Registry
        from nesy.middleware.bus import MessageBus
        bus = Registry.create(MessageBus, "local")
        received = []
        bus.subscribe("test/topic", lambda msg: received.append(msg.payload))
        bus.publish("test/topic", {"data": 42})
        assert len(received) == 1
        assert received[0] == {"data": 42}


# ═══════════════════════════════════════════════════════════════════
# NeSyPlatform DI Integration
# ═══════════════════════════════════════════════════════════════════

class TestNeSyPlatformDI:
    def test_platform_detector_lazy(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p._perception_detector is None
        det = p.detector
        assert det is not None
        assert p._perception_detector is det  # Cached

    def test_platform_extractor_lazy(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p._perception_extractor is None
        ext = p.extractor
        assert ext is not None
        assert p._perception_extractor is ext

    def test_platform_logic_engine_lazy(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p._reasoning_engine is None
        engine = p.logic_engine
        assert engine is not None
        assert p._reasoning_engine is engine

    def test_platform_llm_lazy(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p._llm_provider is None
        llm = p.llm
        assert llm is not None
        assert p._llm_provider is llm

    def test_platform_message_bus_lazy(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        assert p._message_bus is None
        bus = p.message_bus
        assert bus is not None
        assert p._message_bus is bus

    def test_platform_perceive_with_image(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = p.perceive(img)
        assert "detections" in result
        assert "features" in result
        assert "num_objects" in result
        assert result["num_objects"] == 2  # mock returns 2 detections
        assert result["features"]["dim"] == 512

    def test_platform_perceive_non_array_returns_empty(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        result = p.perceive("not an image")
        assert result == {}

    def test_platform_reason_returns_list(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        result = p.reason("nonexistent_relation")
        assert isinstance(result, list)

    def test_platform_reason_with_facts(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        engine = p.logic_engine
        engine.add_relation("on", ["String", "String"])
        engine.add_fact("on", "cup", "table")
        results = p.reason("on")
        assert ("cup", "table") in results

    def test_platform_get_registry_summary(self):
        from nesy import NeSyPlatform
        p = NeSyPlatform()
        # Force plugin loading
        _ = p.detector
        _ = p.logic_engine
        _ = p.llm
        summary = p.get_registry_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        # Should have interfaces registered
        found_detector = False
        for key, plugins in summary.items():
            if "mock" in plugins:
                found_detector = True
        assert found_detector

    def test_platform_config_driven_backend(self):
        """Changing config should change backend."""
        from nesy import NeSyPlatform
        from nesy.core.config import NeSyConfig
        config = NeSyConfig()
        config.perception.object_detection["backend"] = "mock"
        config.perception.object_detection["confidence_threshold"] = 0.9
        p = NeSyPlatform(config=config)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = p.perceive(img)
        # With confidence 0.9, mock returns only chair (0.95) not cup (0.88)
        assert result["num_objects"] == 1

    def test_platform_version(self):
        import nesy
        assert nesy.__version__ == "1.0.0"

    def test_platform_registry_exported(self):
        from nesy import Registry
        assert hasattr(Registry, "list_plugins")
        assert hasattr(Registry, "create")


# ═══════════════════════════════════════════════════════════════════
# ABC Interface Compliance
# ═══════════════════════════════════════════════════════════════════

class TestABCCompliance:
    """Verify that all concrete plugins properly implement their ABCs."""

    def test_mock_detector_is_detector_base(self):
        from nesy.perception.base import ObjectDetectorBase
        from nesy.perception.detectors.mock import MockDetector
        assert issubclass(MockDetector, ObjectDetectorBase)

    def test_yolo_detector_is_detector_base(self):
        from nesy.perception.base import ObjectDetectorBase
        from nesy.perception.detectors.yolo import YOLODetector
        assert issubclass(YOLODetector, ObjectDetectorBase)

    def test_frcnn_detector_is_detector_base(self):
        from nesy.perception.base import ObjectDetectorBase
        from nesy.perception.detectors.faster_rcnn import FasterRCNNDetector
        assert issubclass(FasterRCNNDetector, ObjectDetectorBase)

    def test_mock_extractor_is_extractor_base(self):
        from nesy.perception.base import FeatureExtractorBase
        from nesy.perception.extractors.mock import MockFeatureExtractor
        assert issubclass(MockFeatureExtractor, FeatureExtractorBase)

    def test_clip_extractor_is_extractor_base(self):
        from nesy.perception.base import FeatureExtractorBase
        from nesy.perception.extractors.clip import CLIPExtractor
        assert issubclass(CLIPExtractor, FeatureExtractorBase)

    def test_resnet_extractor_is_extractor_base(self):
        from nesy.perception.base import FeatureExtractorBase
        from nesy.perception.extractors.resnet import ResNetExtractor
        assert issubclass(ResNetExtractor, FeatureExtractorBase)

    def test_dinov2_extractor_is_extractor_base(self):
        from nesy.perception.base import FeatureExtractorBase
        from nesy.perception.extractors.dinov2 import DINOv2Extractor
        assert issubclass(DINOv2Extractor, FeatureExtractorBase)

    def test_scallop_engine_is_logic_base(self):
        from nesy.reasoning.base import LogicEngineBase
        from nesy.reasoning.engines.scallop_plugin import ScallopEngine
        assert issubclass(ScallopEngine, LogicEngineBase)

    def test_minalog_engine_is_logic_base(self):
        from nesy.reasoning.base import LogicEngineBase
        from nesy.reasoning.engines.minalog_plugin import MinalogEngine
        assert issubclass(MinalogEngine, LogicEngineBase)

    def test_pddl_planner_is_planner_base(self):
        from nesy.reasoning.base import PlannerBase
        from nesy.reasoning.engines.pddl_plugin import PDDLPlannerPlugin
        assert issubclass(PDDLPlannerPlugin, PlannerBase)

    def test_mock_llm_is_llm_base(self):
        from nesy.reasoning.llm.base import LLMProviderBase
        from nesy.reasoning.llm.providers.mock import MockLLMProvider
        assert issubclass(MockLLMProvider, LLMProviderBase)

    def test_anthropic_is_llm_base(self):
        from nesy.reasoning.llm.base import LLMProviderBase
        from nesy.reasoning.llm.providers.anthropic_provider import AnthropicProvider
        assert issubclass(AnthropicProvider, LLMProviderBase)

    def test_openai_is_llm_base(self):
        from nesy.reasoning.llm.base import LLMProviderBase
        from nesy.reasoning.llm.providers.openai_provider import OpenAIProvider
        assert issubclass(OpenAIProvider, LLMProviderBase)

    def test_autonomous_agent_is_agent_base(self):
        from nesy.agents.base import AgentBase
        from nesy.agents.plugins import AutonomousAgentPlugin
        assert issubclass(AutonomousAgentPlugin, AgentBase)

    def test_local_bus_is_message_bus(self):
        from nesy.middleware.bus import MessageBus, LocalMessageBus
        assert issubclass(LocalMessageBus, MessageBus)
