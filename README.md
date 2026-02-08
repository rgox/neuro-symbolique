# NeSy: Universal Neuro-Symbolic AI Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Bridging Neural Perception with Symbolic Reasoning for Truly Intelligent AI Systems**

NeSy is a universal platform for building neuro-symbolic AI systems that combine the pattern recognition power of deep learning with the logical reasoning capabilities of symbolic AI. Whether you're building autonomous robots, intelligent agents, or explainable AI systems, NeSy provides the infrastructure to seamlessly integrate neural and symbolic components.

---

## 🎯 Why Neuro-Symbolic AI?

Current AI systems face a fundamental limitation:

- **Pure Neural Systems** (Deep Learning): Excellent at perception and pattern recognition, but opaque, hard to debug, and struggle with logical reasoning and planning.
- **Pure Symbolic Systems** (Logic/Planning): Rigorous and explainable, but fail at sensory grounding—they can't connect abstract symbols to raw sensor data.

**NeSy bridges this gap** by providing:
- ✅ **Zero-Copy Neural-Symbolic Grounding**: Direct memory sharing between neural embeddings and symbolic representations
- ✅ **Differentiable Logic**: Gradient flow through logical rules for end-to-end learning
- ✅ **3D Scene Graphs + Knowledge Graphs**: Unified world representation combining geometric and semantic information
- ✅ **Hardware Abstraction**: Simulated NPU (Neural) / SPU (Symbolic) / CPU architecture for optimal workload distribution
- ✅ **Explainability Built-In**: Logic trace debugger shows *why* the system made each decision

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nesy.git
cd nesy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Hello, NeSy! (10 lines)

```python
from nesy import NeSyPlatform
from PIL import Image

# Initialize platform
platform = NeSyPlatform.from_config('configs/minimal.yaml')

# Perceive the world
image = Image.open('examples/scene.jpg')
platform.perceive(image)

# Reason about what you see
results = platform.reason("find all red cups on tables")
print(results)

# Visualize (opens 3D viewer)
platform.visualize()
```

That's it! You've just built a system that:
1. **Perceives** objects using neural networks
2. **Represents** them in a 3D scene graph
3. **Reasons** about spatial and semantic relationships using logic
4. **Explains** its reasoning with provenance tracking

---

## 🏗️ Architecture Overview

NeSy uses a layered architecture designed for modularity and extensibility:

```
┌─────────────────────────────────────────────┐
│   Applications (Your AI Systems)             │
├─────────────────────────────────────────────┤
│   Dev Tools (Debugger, Viz, Simulation)     │
├─────────────────────────────────────────────┤
│   Reasoning (Logic, Planning, Verification) │
├─────────────────────────────────────────────┤
│   World Model (Scene Graph + KG + VSA)      │
├─────────────────────────────────────────────┤
│   Middleware (Semantic Pub/Sub, ACP, MCP)   │
├─────────────────────────────────────────────┤
│   HAL (NPU/SPU/CPU Simulation)              │
├─────────────────────────────────────────────┤
│   Core Infrastructure (Memory, Scheduling)   │
└─────────────────────────────────────────────┘
```

### Key Components

#### 🧠 **Hardware Abstraction Layer (HAL)**
- **NPU (Neural Processing Unit)**: Simulates specialized hardware for neural network inference
- **SPU (Symbolic Processing Unit)**: Simulates REASON architecture for graph traversal and logic operations
- **UMA (Unified Memory Architecture)**: Zero-copy memory sharing for efficient neural-symbolic grounding

#### 🌍 **World Model**
- **5-Layer Scene Graph**: From dense mesh (L1) to building-level topology (L5)
- **Knowledge Graph**: Ontology-based semantic knowledge (T-Box + A-Box)
- **VSA (Vector-Symbolic Architectures)**: Holographic representations for binding neural and symbolic spaces

#### 🤔 **Reasoning Engine**
- **Differentiable Logic**: Scallop/DeepProbLog integration for gradient-based learning
- **TAMP/STAMP Planning**: Task and motion planning for robotics
- **LLM Dispatcher**: Natural language to formal specifications
- **Formal Verification**: Barrier certificates and temporal logic for safety

#### 🔧 **Development Tools**
- **Logic Trace Debugger**: Visualize rule activations and provenance
- **Semantic Simulator**: Test logic without running full perception
- **3D Scene Viewer**: Rerun-based visualization of scene graphs

---

## 📚 Examples

We provide progressive examples from simple to complex:

### 1. [Minimal Perception](examples/01_minimal_perception/)
Basic object detection → scene graph pipeline

### 2. [Scene Graph Basics](examples/02_scene_graph_basics/)
Constructing and querying 3D scene representations

### 3. [Simple Reasoning](examples/03_simple_reasoning/)
Writing logic rules and querying the knowledge base

### 4. [Neural-Symbolic Grounding](examples/04_grounding/)
VSA encoding and zero-copy memory sharing

### 5. [End-to-End Integration](examples/05_end_to_end/)
Full pipeline with debugging and visualization

---

## 🎓 Use Cases

NeSy is designed for researchers and practitioners working on:

### 🤖 **Autonomous Robotics**
- Mobile robot navigation with semantic understanding
- Robotic manipulation with explainable grasping
- Multi-robot coordination with agent communication

### 🏥 **Safety-Critical AI**
- Medical diagnosis systems with explainable reasoning
- Autonomous vehicles with formal verification
- Industrial automation with provable safety

### 🔬 **AI Research**
- Novel neuro-symbolic architectures
- Explainable AI (XAI) systems
- Hybrid learning algorithms

### 🎮 **Intelligent Agents**
- Game AI with strategic reasoning
- Virtual assistants with common-sense understanding
- Interactive learning systems

---

## 📖 Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 10 minutes
- **[Tutorials](docs/tutorials/)** - Step-by-step guides for common tasks
- **[API Reference](docs/api/)** - Complete API documentation
- **[Architecture Deep Dive](docs/architecture/)** - System design and implementation details

---

## 🗺️ Roadmap

### ✅ MVP (Current - Q2 2026)
- [x] Core infrastructure (UMA, config, logging)
- [x] HAL simulation (NPU/SPU/CPU)
- [x] 3-layer scene graph (L1-L3)
- [ ] Scallop logic integration
- [ ] Basic perception pipeline
- [ ] VSA grounding layer
- [ ] Logic trace debugger
- [ ] 5 working examples

### 🚧 Phase 2 (Q3 2026)
- TAMP/STAMP planning
- LLM dispatcher (MCP integration)
- DeepProbLog support
- Temporal reasoning
- Verification framework

### 🔮 Phase 3 (Q4 2026)
- ROS2 semantic bridge
- IEEE 1872.2 ontology support
- Agent Communication Protocol (ACP)
- Multi-agent coordination

### 🎯 Phase 4 (Q1 2027)
- C++ performance optimization
- Full 5-layer scene graph (L4-L5)
- Procedural scene generation
- Advanced debugger features
- Production deployment tools

---

## 🤝 Contributing

We welcome contributions! Whether it's:
- 🐛 Bug reports
- 💡 Feature requests
- 📝 Documentation improvements
- 🔧 Code contributions

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

NeSy builds upon cutting-edge research in neuro-symbolic AI:

- **Scallop**: Differentiable logic programming (Li et al., 2023)
- **Hydra**: Dynamic scene graphs (Hughes et al., 2022)
- **REASON**: Symbolic processing architecture (Liang et al., 2024)
- **IEEE 1872.2**: Robotics ontology standards
- **Vector-Symbolic Architectures**: Hyperdimensional computing (Kanerva, 2009)

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/nesy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nesy/discussions)

---

<p align="center">
  <strong>Build AI systems that think <em>and</em> perceive. Start with NeSy.</strong>
</p>
