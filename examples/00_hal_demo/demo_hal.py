"""
HAL Demo: Neural-Symbolic Processing Units in Action

This demo shows how the Hardware Abstraction Layer works:
- NPU for neural computations (PyTorch wrapper)
- SPU for symbolic reasoning (graph traversal, logic)
- CPU orchestrator for workload distribution
- UMA for zero-copy memory sharing
"""

import torch
import torch.nn as nn
from nesy import NeSyPlatform
from nesy.hal import Workload, WorkloadType
from nesy.core.memory import DataType


def main():
    print("=" * 60)
    print("NeSy Platform - HAL Demonstration")
    print("=" * 60)

    # Initialize platform
    print("\n1. Initializing NeSy Platform...")
    platform = NeSyPlatform.from_config("../../configs/minimal.yaml")
    print(f"   ✓ Platform initialized")
    print(f"   ✓ UMA device: {platform.uma.device}")

    # Show HAL capabilities
    print("\n2. Hardware Abstraction Layer Capabilities:")
    print(f"\n   NPU (Neural Processing Unit):")
    npu_caps = platform.hal.npu.get_capabilities()
    for key, value in npu_caps.items():
        print(f"     - {key}: {value}")

    print(f"\n   SPU (Symbolic Processing Unit):")
    spu_caps = platform.hal.spu.get_capabilities()
    for key, value in spu_caps.items():
        print(f"     - {key}: {value}")

    # Demo 1: NPU Neural Computation
    print("\n" + "=" * 60)
    print("DEMO 1: NPU - Neural Network Inference")
    print("=" * 60)

    # Create a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    model = SimpleNet()
    input_tensor = torch.randn(1, 10)  # Batch of 1, 10 features

    print("\n   Model: SimpleNet (10 -> 20 -> 5)")
    print(f"   Input shape: {input_tensor.shape}")

    # Execute on NPU
    npu_workload = Workload(
        workload_type=WorkloadType.NEURAL_INFERENCE,
        operation=model,
        inputs={"input": input_tensor},
        outputs=["neural_output"],
    )

    result = platform.hal.npu.execute(npu_workload)

    print(f"\n   ✓ Execution successful: {result.success}")
    print(f"   ✓ Duration: {result.duration_ms:.2f} ms")
    print(f"   ✓ Memory used: {result.memory_used_bytes / 1024:.2f} KB")
    print(f"   ✓ Output stored in UMA: {result.outputs['output_keys']}")

    # Demo 2: SPU Symbolic Reasoning
    print("\n" + "=" * 60)
    print("DEMO 2: SPU - Graph Traversal and Symbolic Reasoning")
    print("=" * 60)

    # Build a knowledge graph
    from nesy.hal import Graph

    graph = Graph()

    # Add nodes (concepts/entities)
    print("\n   Building knowledge graph:")
    entities = ["cup", "table", "robot", "kitchen", "living_room"]
    for entity in entities:
        graph.add_node(entity, {"type": "object" if entity not in ["kitchen", "living_room"] else "place"})
        print(f"     - Added node: {entity}")

    # Add edges (relations)
    relations = [
        ("cup", "on", "table"),
        ("table", "in", "kitchen"),
        ("robot", "in", "kitchen"),
        ("kitchen", "connected_to", "living_room"),
    ]

    print("\n   Adding relations:")
    for src, rel, dst in relations:
        graph.add_edge(src, rel, dst)
        print(f"     - {src} --[{rel}]--> {dst}")

    # Add graph to SPU
    platform.hal.spu.add_graph("scene_graph", graph)

    # Demo 2a: Graph Traversal (BFS)
    print("\n   Executing BFS from 'robot':")
    traversal_workload = Workload(
        workload_type=WorkloadType.GRAPH_TRAVERSAL,
        operation="bfs",
        inputs={"graph": "scene_graph", "start": "robot"},
    )

    result = platform.hal.spu.execute(traversal_workload)
    print(f"     ✓ Nodes reachable from robot: {result.outputs['path']}")
    print(f"     ✓ Duration: {result.duration_ms:.2f} ms")

    # Demo 2b: Add facts and query
    print("\n   Adding symbolic facts:")
    facts = [
        ("cup", "color", "red"),
        ("cup", "contains", "coffee"),
        ("table", "material", "wood"),
    ]

    for fact in facts:
        platform.hal.spu.facts.add(fact)
        print(f"     - Added: {fact}")

    # Query facts
    print("\n   Querying: What is on the table?")
    query_workload = Workload(
        workload_type=WorkloadType.LOGIC_EVALUATION,
        operation="query_fact",
        inputs={"pattern": (None, "on", "table")},  # Wildcard query
    )

    result = platform.hal.spu.execute(query_workload)
    print(f"     ✓ Results: {result.outputs['results']}")
    print(f"     ✓ Duration: {result.duration_ms:.2f} ms")

    # Demo 3: CPU Orchestration
    print("\n" + "=" * 60)
    print("DEMO 3: CPU Orchestrator - Workload Routing")
    print("=" * 60)

    print("\n   Testing automatic workload routing...")

    # Neural workload -> should route to NPU
    neural_wl = Workload(
        workload_type=WorkloadType.NEURAL_INFERENCE,
        operation=lambda x: x * 2,
        inputs={"x": torch.tensor([1.0, 2.0, 3.0])},
    )

    # Symbolic workload -> should route to SPU
    symbolic_wl = Workload(
        workload_type=WorkloadType.GRAPH_TRAVERSAL,
        operation="neighbors",
        inputs={"graph": "scene_graph", "start": "cup"},
    )

    print("\n   Executing neural workload via orchestrator:")
    result1 = platform.hal.cpu.execute(neural_wl)
    print(f"     ✓ Routed to: {result1.device_type.value}")
    print(f"     ✓ Success: {result1.success}")

    print("\n   Executing symbolic workload via orchestrator:")
    result2 = platform.hal.cpu.execute(symbolic_wl)
    print(f"     ✓ Routed to: {result2.device_type.value}")
    print(f"     ✓ Result: {result2.outputs}")

    # Demo 4: Zero-Copy Grounding
    print("\n" + "=" * 60)
    print("DEMO 4: UMA - Zero-Copy Neural-Symbolic Grounding")
    print("=" * 60)

    print("\n   Demonstrating zero-copy memory sharing...")

    # NPU creates embeddings
    embeddings = torch.randn(5, 128)  # 5 objects, 128-dim embeddings
    buffer = platform.uma.allocate(
        key="object_embeddings",
        shape=(5, 128),
        dtype=DataType.FLOAT32,
        device=platform.hal.npu.device_type,
    )
    buffer.data = embeddings

    print(f"     ✓ NPU created embeddings: {buffer.shape}")
    print(f"     ✓ Memory location: {buffer.data.data_ptr()}")

    # Annotate with symbolic labels (grounding)
    platform.uma.annotate(
        "object_embeddings",
        {
            "objects": ["cup", "table", "robot", "chair", "lamp"],
            "semantic_type": "visual_embedding",
        },
    )

    print(f"     ✓ Added semantic annotations")

    # SPU accesses the same embeddings (zero-copy)
    transferred = platform.uma.zero_copy_transfer(
        "object_embeddings",
        platform.hal.npu.device_type,
        platform.hal.spu.device_type,
    )

    print(f"     ✓ SPU accessed embeddings via zero-copy transfer")
    print(f"     ✓ Same memory location: {transferred.data.data_ptr()}")
    print(f"     ✓ Semantic labels: {transferred.metadata['objects']}")

    # Performance Statistics
    print("\n" + "=" * 60)
    print("Performance Statistics")
    print("=" * 60)

    stats = platform.hal.cpu.get_device_statistics()
    for device_type, device_stats in stats.items():
        print(f"\n   {device_type.upper()}:")
        for key, value in device_stats.items():
            if isinstance(value, float):
                print(f"     - {key}: {value:.2f}")
            else:
                print(f"     - {key}: {value}")

    print("\n" + "=" * 60)
    print("HAL Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  ✓ NPU handles neural computations efficiently")
    print("  ✓ SPU handles symbolic reasoning and graph traversal")
    print("  ✓ CPU orchestrates workload distribution automatically")
    print("  ✓ UMA enables zero-copy neural-symbolic grounding")
    print("  ✓ All components integrated in unified platform")


if __name__ == "__main__":
    main()
