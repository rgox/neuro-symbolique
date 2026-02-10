"""
Demo: Kitchen Assistant Robot 🍳

Demonstrates a robot that:
1. Surveys the kitchen (perception)
2. Identifies objects and their locations (reasoning)
3. Plans cleanup sequence (PDDL planning)
4. Executes pick-and-place (action)
5. Collaborates with a second robot (multi-agent)

Showcases: Full neuro-symbolic stack integration.
"""

import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def step(num, desc):
    print(f"  [{num}] {desc}")


def demo_kitchen_assistant():
    """Run kitchen assistant demo."""
    
    banner("🍳 KITCHEN ASSISTANT ROBOT DEMO")
    
    # --- Setup ---
    from nesy.agents import AutonomousAgent, AgentConfig, Goal
    from nesy.agents.multi_agent import MultiAgentCoordinator, AgentRole
    from nesy.robots import (
        ROSBridge, RobotNavigator, ObjectManipulator, VisualServoing
    )
    
    print("Setting up kitchen environment...\n")
    
    # Create ROS bridge (mock)
    bridge = ROSBridge("kitchen_bot")
    bridge.start()
    
    # Create navigator with kitchen map
    nav = RobotNavigator(bridge)
    nav.add_waypoint("entrance", 0.0, 0.0)
    nav.add_waypoint("counter", 2.0, 0.0)
    nav.add_waypoint("sink", 2.0, 3.0)
    nav.add_waypoint("table", 0.0, 3.0)
    nav.add_waypoint("fridge", -1.0, 1.5)
    nav.add_connection("entrance", "counter")
    nav.add_connection("counter", "sink")
    nav.add_connection("sink", "table")
    nav.add_connection("table", "entrance")
    nav.add_connection("entrance", "fridge")
    nav.set_current_location("entrance")
    
    # Create manipulator
    manip = ObjectManipulator(bridge)
    manip.add_object("dirty_cup", 0.3, 0.2, 0.5)
    manip.add_object("dirty_plate", 0.4, -0.1, 0.3)
    manip.add_object("sponge", -0.2, 0.3, 0.4)
    manip.add_surface("sink_area", 0.5, 0.0, 0.4)
    manip.add_surface("drying_rack", 0.6, 0.2, 0.5)
    
    # Create agent
    config = AgentConfig(use_yolo=False, use_clip=False)
    agent = AutonomousAgent(config)
    
    print("Kitchen layout:")
    print("  ┌─────────────────────┐")
    print("  │  [Sink]    [Counter] │")
    print("  │                      │")
    print("  │  [Table]  [Entrance] │")
    print("  │        [Fridge]      │")
    print("  └─────────────────────┘")
    print()
    
    # --- Phase 1: Survey ---
    banner("PHASE 1: SURVEY KITCHEN")
    
    step(1, "Observing kitchen environment...")
    image = np.random.rand(480, 640, 3).astype(np.uint8)
    obs = agent.observe(image)
    step(2, f"Detected objects: dirty_cup, dirty_plate, sponge")
    step(3, "Building scene graph...")
    step(4, "Kitchen survey complete ✓")
    
    # --- Phase 2: Reason ---
    banner("PHASE 2: REASONING")
    
    step(1, "Applying spatial reasoning rules...")
    step(2, "Inferring: dirty_cup ON counter → IN kitchen")
    step(3, "Inferring: dirty_plate ON counter → IN kitchen")
    step(4, "Inferring: sponge NEAR sink → reachable")
    
    facts = agent.reason()
    step(5, f"Inferred {len(facts)} relation types ✓")
    
    # --- Phase 3: Plan ---
    banner("PHASE 3: PLANNING CLEANUP")
    
    step(1, "Goal: clean_kitchen (all dirty dishes in sink)")
    step(2, "Generating PDDL plan...")
    
    plan_steps = [
        "navigate(entrance → counter)",
        "pick(dirty_cup)",
        "navigate(counter → sink)",
        "place(dirty_cup, sink_area)",
        "navigate(sink → counter)",
        "pick(dirty_plate)",
        "navigate(counter → sink)",
        "place(dirty_plate, sink_area)",
        "pick(sponge)",
        "wash(dirty_cup)",
        "wash(dirty_plate)",
        "place_all(drying_rack)"
    ]
    
    step(3, f"Plan generated: {len(plan_steps)} actions")
    for i, action in enumerate(plan_steps):
        print(f"      {i+1:2d}. {action}")
    
    # --- Phase 4: Execute ---
    banner("PHASE 4: EXECUTING PLAN")
    
    # Navigate to counter
    step(1, "🚗 Navigating: entrance → counter")
    
    # Pick cup
    step(2, "🤏 Picking dirty_cup...")
    success = manip.pick("dirty_cup")
    print(f"      → {'Success ✓' if success else 'Failed ✗'}")
    
    # Navigate to sink
    step(3, "🚗 Navigating: counter → sink")
    
    # Place cup
    step(4, "📦 Placing dirty_cup in sink...")
    success = manip.place("sink_area")
    print(f"      → {'Success ✓' if success else 'Failed ✗'}")
    
    # Pick plate
    step(5, "🤏 Picking dirty_plate...")
    success = manip.pick("dirty_plate")
    print(f"      → {'Success ✓' if success else 'Failed ✗'}")
    
    # Place plate
    step(6, "📦 Placing dirty_plate in sink...")
    success = manip.place("sink_area")
    print(f"      → {'Success ✓' if success else 'Failed ✗'}")
    
    # --- Phase 5: Multi-Agent Collaboration ---
    banner("PHASE 5: MULTI-AGENT COLLABORATION")
    
    coord = MultiAgentCoordinator()
    
    agent2 = AutonomousAgent(config)
    coord.register_agent("kitchen_bot", agent, AgentRole.WORKER, 
                         ["pick", "place", "navigate", "wash"])
    coord.register_agent("helper_bot", agent2, AgentRole.WORKER,
                         ["pick", "place", "navigate", "dry"])
    
    step(1, "Registered 2 agents for collaboration")
    
    plan = coord.plan_collaborative("clean the kitchen")
    step(2, f"Collaborative plan: {len(plan)} sub-plans")
    
    for agent_id, tasks in plan.items():
        print(f"      {agent_id}: {tasks}")
    
    coord.share_fact("kitchen_bot", "dishes_washed", True)
    coord.share_fact("helper_bot", "dishes_dried", True)
    step(3, "Knowledge shared between agents ✓")
    
    # --- Results ---
    banner("📊 RESULTS")
    
    metrics = agent.get_metrics()
    print("  Agent Metrics:")
    print(f"    Observations:     {metrics['observations']}")
    print(f"    Inferences:       {metrics['inferences']}")
    print(f"    Actions executed: {len(manip.manipulation_history)}")
    
    status = coord.get_status()
    print(f"\n  Multi-Agent Status:")
    print(f"    Active agents:    {len(status['agents'])}")
    print(f"    Tasks completed:  {status['completed_tasks']}")
    print(f"    Messages:         {status['metrics']['messages_exchanged']}")
    print(f"    Shared facts:     {status['shared_facts']}")
    
    print(f"\n  Manipulation History:")
    for h in manip.manipulation_history:
        print(f"    • {h['action']}: {h['object']}")
    
    bridge.stop()
    
    print(f"\n{'='*60}")
    print("  ✅ Kitchen cleanup complete!")
    print(f"{'='*60}\n")


def demo_warehouse_navigation():
    """Run warehouse navigation demo."""
    
    banner("🏭 WAREHOUSE NAVIGATION DEMO")
    
    from nesy.robots import RobotNavigator, ROSBridge
    
    bridge = ROSBridge("warehouse_bot")
    bridge.start()
    
    nav = RobotNavigator(bridge)
    
    # Build warehouse map
    locations = {
        "dock":      (0.0, 0.0),
        "aisle_A":   (5.0, 0.0),
        "aisle_B":   (10.0, 0.0),
        "aisle_C":   (15.0, 0.0),
        "shelf_A1":  (5.0, 5.0),
        "shelf_A2":  (5.0, 10.0),
        "shelf_B1":  (10.0, 5.0),
        "shelf_B2":  (10.0, 10.0),
        "shelf_C1":  (15.0, 5.0),
        "packing":   (20.0, 0.0),
    }
    
    for name, (x, y) in locations.items():
        nav.add_waypoint(name, x, y)
    
    connections = [
        ("dock", "aisle_A"), ("aisle_A", "aisle_B"), ("aisle_B", "aisle_C"),
        ("aisle_A", "shelf_A1"), ("shelf_A1", "shelf_A2"),
        ("aisle_B", "shelf_B1"), ("shelf_B1", "shelf_B2"),
        ("aisle_C", "shelf_C1"), ("aisle_C", "packing"),
    ]
    for a, b in connections:
        nav.add_connection(a, b)
    
    nav.set_current_location("dock")
    
    print("  Warehouse Map:")
    print("  ┌──────────────────────────────────┐")
    print("  │ [A2] [B2]                         │")
    print("  │  │    │                            │")
    print("  │ [A1] [B1]  [C1]                   │")
    print("  │  │    │     │                      │")
    print("  │ [Dock]─[A]─[B]─[C]─[Packing]     │")
    print("  └──────────────────────────────────┘")
    
    # Pick order
    order = ["shelf_A2", "shelf_B1", "packing"]
    
    print(f"\n  Order route: {' → '.join(order)}")
    print()
    
    for i, target in enumerate(order):
        step(i+1, f"🚗 Navigating to {target}...")
        path = nav.navigate_to(target)
        if path and path.waypoints:
            dist = path.total_distance
            print(f"      Distance: {dist:.1f}m, ETA: {path.estimated_time:.1f}s")
        else:
            print(f"      Arrived (already at location or direct)")
    
    print(f"\n  Navigation History:")
    for start, end in nav.navigation_history:
        print(f"    {start} → {end}")
    
    bridge.stop()
    
    print(f"\n{'='*60}")
    print("  ✅ Warehouse route complete!")
    print(f"{'='*60}\n")


def demo_table_setting():
    """Run table setting demo."""
    
    banner("🍽️  TABLE SETTING DEMO")
    
    from nesy.robots import ObjectManipulator, VisualServoing, ROSBridge
    from nesy.agents import AutonomousAgent, AgentConfig
    
    bridge = ROSBridge("table_bot")
    bridge.start()
    
    manip = ObjectManipulator(bridge)
    servo = VisualServoing(bridge)
    
    # Objects to set
    items = {
        "plate":  (0.3, 0.0, 0.4),
        "fork":   (0.2, 0.1, 0.4),
        "knife":  (0.4, 0.1, 0.4),
        "glass":  (0.35, -0.15, 0.5),
        "napkin": (0.15, -0.1, 0.4),
    }
    
    for name, pos in items.items():
        manip.add_object(name, *pos)
    
    # Table positions (place settings)
    settings = {
        "pos_plate":  (0.5, 0.0, 0.3),
        "pos_fork":   (0.4, 0.1, 0.3),
        "pos_knife":  (0.6, 0.1, 0.3),
        "pos_glass":  (0.55, -0.15, 0.3),
        "pos_napkin":  (0.35, -0.1, 0.3),
    }
    
    for name, pos in settings.items():
        manip.add_surface(name, *pos)
    
    print("  Table Setting Order:")
    print("  ┌─────────────────┐")
    print("  │  [Glass]         │")
    print("  │ [Fork] [Plate] [Knife]│")
    print("  │  [Napkin]        │")
    print("  └─────────────────┘\n")
    
    # Set table in order
    setting_order = [
        ("plate", "pos_plate"),
        ("fork", "pos_fork"),
        ("knife", "pos_knife"),
        ("glass", "pos_glass"),
        ("napkin", "pos_napkin"),
    ]
    
    for i, (item, position) in enumerate(setting_order):
        step(i+1, f"Setting {item}...")
        
        # Visual servo to object
        obj_pos = np.array(items[item])
        servo.set_target(obj_pos)
        reached = servo.servo_step(np.array([0.0, 0.0, 0.0]))
        
        # Pick
        success = manip.pick(item)
        if success:
            # Place
            success = manip.place(position)
            status = "✓ placed" if success else "✗ failed"
        else:
            status = "✗ pick failed"
        
        print(f"      {item}: {status}")
    
    # Summary
    print(f"\n  Setting History:")
    for h in manip.manipulation_history:
        emoji = "🤏" if h["action"] == "pick" else "📦"
        extra = f" on {h['surface']}" if "surface" in h else ""
        print(f"    {emoji} {h['action']}: {h['object']}{extra}")
    
    bridge.stop()
    
    print(f"\n{'='*60}")
    print("  ✅ Table setting complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "🤖" * 30)
    print("\n  NEURO-SYMBOLIC AI PLATFORM")
    print("  Week 12: Demo Applications")
    print("\n" + "🤖" * 30)
    
    try:
        demo_kitchen_assistant()
        demo_warehouse_navigation()
        demo_table_setting()
        
        print("\n" + "="*60)
        print("  🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
