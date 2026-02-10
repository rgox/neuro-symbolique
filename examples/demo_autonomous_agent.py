"""
Autonomous Agent Demo.

Demonstrates complete perception-reasoning-planning-action loop.
"""

import cv2
import numpy as np
import time
from pathlib import Path

from nesy.agents import AutonomousAgent, AgentConfig, Goal


def create_mock_camera():
    """Create mock camera that generates test images."""
    def get_frame():
        # Generate synthetic scene
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw background
        img[:] = (50, 50, 50)
        
        # Draw "table"
        cv2.rectangle(img, (100, 300), (540, 400), (139, 69, 19), -1)
        
        # Draw "objects"
        cv2.circle(img, (200, 250), 30, (0, 0, 255), -1)  # Red cup
        cv2.circle(img, (400, 250), 30, (0, 255, 0), -1)  # Green cup
        
        return img
    
    return get_frame


def demo_single_step():
    """Demo: Single step execution."""
    print("=" * 60)
    print("DEMO 1: Single Step Execution")
    print("=" * 60)
    
    # Create agent
    config = AgentConfig(
        use_yolo=False,  # Use mock perception for demo
        use_clip=False,
        loop_frequency=1.0
    )
    agent = AutonomousAgent(config)
    
    # Get camera
    camera = create_mock_camera()
    
    # Step 1: PERCEIVE
    print("\n1. PERCEIVING...")
    image = camera()
    obs = agent.observe(image)
    print(f"   Observed {len(obs.detections)} objects")
    
    # Step 2: REASON
    print("\n2. REASONING...")
    facts = agent.reason()
    print(f"   Inferred facts: {facts}")
    
    # Step 3: PLAN
    print("\n3. PLANNING...")
    goal = Goal(
        description="Clean the table",
        target_predicates=["clear(table)"],
        priority=1.0
    )
    plan = agent.plan(goal)
    
    if plan:
        print(f"   Generated plan with {len(plan)} actions:")
        for i, action in enumerate(plan):
            print(f"     {i+1}. {action}")
    else:
        print("   No plan found")
    
    # Step 4: EXECUTE (simulated)
    if plan:
        print("\n4. EXECUTING...")
        print("   (Simulated execution)")
        for i, action in enumerate(plan):
            print(f"   Executing: {action}")
            time.sleep(0.5)
    
    # Metrics
    print("\n" + "=" * 60)
    print("METRICS:")
    metrics = agent.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def demo_autonomous_loop():
    """Demo: Autonomous loop."""
    print("\n\n" + "=" * 60)
    print("DEMO 2: Autonomous Loop (3 iterations)")
    print("=" * 60)
    
    config = AgentConfig(
        use_yolo=False,
        use_clip=False,
        loop_frequency=2.0  # 2 Hz
    )
    agent = AutonomousAgent(config)
    
    camera = create_mock_camera()
    
    goal = Goal(
        description="Monitor scene",
        target_predicates=["monitored()"],
        priority=1.0
    )
    
    print("\nStarting autonomous loop...")
    print("Press Ctrl+C to stop\n")
    
    # Simple loop (not using agent.run() for demo)
    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # PERCEIVE
        image = camera()
        obs = agent.observe(image)
        print(f"Observed: {len(obs.detections)} objects")
        
        # REASON
        facts = agent.reason()
        print(f"Reasoned: {len(facts)} fact types")
        
        # PLAN (optional)
        # plan = agent.plan(goal)
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("Final Metrics:")
    for key, value in agent.get_metrics().items():
        print(f"  {key}: {value}")
    print("=" * 60)


def demo_state_transitions():
    """Demo: Agent state transitions."""
    print("\n\n" + "=" * 60)
    print("DEMO 3: State Transitions")
    print("=" * 60)
    
    config = AgentConfig(use_yolo=False, use_clip=False)
    agent = AutonomousAgent(config)
    
    camera = create_mock_camera()
    
    states = []
    
    # Track state transitions
    print(f"\nInitial state: {agent.state.value}")
    states.append(agent.state.value)
    
    # Observe
    agent.observe(camera())
    print(f"After observe: {agent.state.value}")
    states.append(agent.state.value)
    
    # Reason
    agent.reason()
    print(f"After reason: {agent.state.value}")
    states.append(agent.state.value)
    
    # Plan
    goal = Goal("test", ["target()"])
    agent.plan(goal)
    print(f"After plan: {agent.state.value}")
    states.append(agent.state.value)
    
    print(f"\nState transition sequence: {' → '.join(states)}")
    print("=" * 60)


if __name__ == "__main__":
    print("\n🤖 AUTONOMOUS AGENT DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Run demos
        demo_single_step()
        demo_autonomous_loop()
        demo_state_transitions()
        
        print("\n✅ All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
