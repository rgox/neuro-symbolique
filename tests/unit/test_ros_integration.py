"""
Tests for ROS Integration.

Tests ROS bridge, navigation, manipulation, and visual servoing.
All tests work without ROS installed (mock mode).
"""

import pytest
import numpy as np

from nesy.robots import (
    ROSBridge,
    ROSNodeState,
    SensorData,
    RobotNavigator,
    TopologicalMap,
    ObjectManipulator,
    VisualServoing,
    Waypoint,
)


class TestROSBridge:
    """Test ROS bridge."""
    
    @pytest.fixture
    def bridge(self):
        """Create mock ROS bridge."""
        b = ROSBridge("test_node")
        b.start()
        yield b
        b.stop()
    
    def test_start_stop(self, bridge):
        """Test node lifecycle."""
        assert bridge.state == ROSNodeState.ACTIVE
        bridge.stop()
        assert bridge.state == ROSNodeState.FINALIZED
    
    def test_subscribe(self, bridge):
        """Test topic subscription."""
        callback = lambda msg: None
        bridge.subscribe("/test_topic", "std_msgs/String", callback)
        
        assert "/test_topic" in bridge.subscribers
    
    def test_publish_velocity(self, bridge):
        """Test velocity publishing."""
        bridge.publish_velocity(linear=0.5, angular=0.1)
        
        assert bridge.metrics["messages_sent"] == 1
    
    def test_publish_pose(self, bridge):
        """Test pose publishing."""
        bridge.publish_pose(1.0, 2.0, 0.0)
        
        assert bridge.metrics["messages_sent"] == 1
    
    def test_sensor_buffering(self, bridge):
        """Test sensor data buffering."""
        sensor = SensorData(
            timestamp=1.0,
            sensor_type="camera",
            data=np.zeros((480, 640, 3))
        )
        bridge._buffer_sensor(sensor)
        
        latest = bridge.get_latest_sensor("camera")
        assert latest is not None
        assert latest.sensor_type == "camera"
    
    def test_metrics(self, bridge):
        """Test metrics."""
        metrics = bridge.get_metrics()
        
        assert "messages_received" in metrics
        assert "messages_sent" in metrics
        assert metrics["state"] == "active"


class TestTopologicalMap:
    """Test topological map."""
    
    @pytest.fixture
    def topo_map(self):
        """Create test map."""
        m = TopologicalMap()
        m.add_waypoint("kitchen", 0.0, 0.0)
        m.add_waypoint("hallway", 3.0, 0.0)
        m.add_waypoint("bedroom", 3.0, 4.0)
        m.add_connection("kitchen", "hallway")
        m.add_connection("hallway", "bedroom")
        return m
    
    def test_waypoints(self, topo_map):
        """Test waypoint management."""
        assert len(topo_map.waypoints) == 3
        assert "kitchen" in topo_map.waypoints
    
    def test_connections(self, topo_map):
        """Test connections."""
        neighbors = topo_map.get_neighbors("kitchen")
        assert "hallway" in neighbors
    
    def test_distance(self, topo_map):
        """Test distance calculation."""
        dist = topo_map.distance("kitchen", "hallway")
        assert abs(dist - 3.0) < 0.01
    
    def test_pddl_facts(self, topo_map):
        """Test PDDL fact generation."""
        facts = topo_map.to_pddl_facts()
        
        assert any("location(kitchen)" in f for f in facts)
        assert any("connected" in f for f in facts)


class TestRobotNavigator:
    """Test robot navigator."""
    
    @pytest.fixture
    def navigator(self):
        """Create navigator with test map."""
        nav = RobotNavigator()
        nav.add_waypoint("A", 0.0, 0.0)
        nav.add_waypoint("B", 3.0, 0.0)
        nav.add_waypoint("C", 3.0, 4.0)
        nav.add_connection("A", "B")
        nav.add_connection("B", "C")
        nav.set_current_location("A")
        return nav
    
    def test_set_location(self, navigator):
        """Test setting current location."""
        assert navigator.current_location == "A"
    
    def test_navigate_same_location(self, navigator):
        """Test navigating to current location."""
        path = navigator.navigate_to("A")
        assert path is not None
        assert len(path.waypoints) == 0


class TestObjectManipulator:
    """Test object manipulator."""
    
    @pytest.fixture
    def manipulator(self):
        """Create manipulator."""
        m = ObjectManipulator()
        m.add_object("cup", 0.3, 0.2, 0.5)
        m.add_object("plate", 0.4, -0.1, 0.3)
        m.add_surface("table", 0.5, 0.0, 0.4)
        return m
    
    def test_pick(self, manipulator):
        """Test picking object."""
        success = manipulator.pick("cup")
        
        assert success
        assert manipulator.holding == "cup"
    
    def test_pick_already_holding(self, manipulator):
        """Test picking while holding."""
        manipulator.pick("cup")
        success = manipulator.pick("plate")
        
        assert not success
    
    def test_place(self, manipulator):
        """Test placing object."""
        manipulator.pick("cup")
        success = manipulator.place("table")
        
        assert success
        assert manipulator.holding is None
    
    def test_place_not_holding(self, manipulator):
        """Test placing when not holding."""
        success = manipulator.place("table")
        
        assert not success
    
    def test_pick_unreachable(self, manipulator):
        """Test picking unreachable object."""
        manipulator.add_object("far_obj", 5.0, 5.0, 0.5)
        success = manipulator.pick("far_obj")
        
        assert not success


class TestVisualServoing:
    """Test visual servoing."""
    
    @pytest.fixture
    def servo(self):
        """Create servo controller."""
        return VisualServoing()
    
    def test_set_target(self, servo):
        """Test setting target."""
        target = np.array([1.0, 2.0, 0.5])
        servo.set_target(target)
        
        assert servo.target_position is not None
    
    def test_compute_control(self, servo):
        """Test control computation."""
        servo.set_target(np.array([1.0, 0.0, 0.0]))
        
        current = np.array([0.0, 0.0, 0.0])
        linear, angular = servo.compute_control(current)
        
        assert linear > 0  # Should move toward target
    
    def test_at_target(self, servo):
        """Test behavior at target."""
        target = np.array([1.0, 0.0, 0.0])
        servo.set_target(target)
        
        # Already at target
        linear, angular = servo.compute_control(target)
        
        assert linear == 0.0
        assert angular == 0.0
    
    def test_servo_step(self, servo):
        """Test servo step."""
        servo.set_target(np.array([1.0, 0.0, 0.0]))
        
        # At target
        reached = servo.servo_step(np.array([1.0, 0.0, 0.0]))
        assert reached
        
        # Not at target
        servo.set_target(np.array([1.0, 0.0, 0.0]))
        reached = servo.servo_step(np.array([0.0, 0.0, 0.0]))
        assert not reached
