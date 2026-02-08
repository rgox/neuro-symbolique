"""
Perception Pipeline Demo.

This demo showcases the full perception pipeline:
1. Object detection from images
2. Visual feature extraction
3. Automatic scene graph updates
4. Embedding storage in UMA
5. Spatial relation computation

The demo simulates a robot observing a kitchen scene and building
a 3D scene graph from visual observations.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nesy import NeSyPlatform
from nesy.perception import PerceptionPipeline
from nesy.world_model import SceneGraph, ObjectLayer, LayerType


def create_mock_image(size=(480, 640)):
    """Create a mock image for demonstration."""
    h, w = size
    
    # Create RGB image with gradient
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(h):
        image[i, :, 0] = int(255 * i / h)  # Red gradient
        image[i, :, 2] = int(255 * (1 - i / h))  # Blue gradient
    
    # Add some "objects" (colored rectangles)
    # Table (brown-ish rectangle)
    image[300:400, 200:400] = [139, 69, 19]
    
    # Cup (red rectangle)
    image[250:320, 280:340] = [255, 0, 0]
    
    return image


def main():
    print("=" * 70)
    print("PERCEPTION PIPELINE - DEMONSTRATION")
    print("=" * 70)
    
    # Initialize platform
    print("\n1. Initializing NeSy Platform...")
    platform = NeSyPlatform.from_config("../../configs/minimal.yaml")
    print(f"   ✓ Platform initialized")
    print(f"   ✓ UMA available: {platform.uma is not None}")
    print(f"   ✓ HAL available: {platform.hal is not None}")
    
    # Create scene graph
    print("\n2. Creating Scene Graph...")
    scene_graph = SceneGraph(uma=platform.uma, logger=platform.logger)
    print(f"   ✓ Scene graph created")
    
    # Create perception pipeline
    print("\n3. Creating Perception Pipeline...")
    pipeline = PerceptionPipeline(
        scene_graph=scene_graph,
        uma=platform.uma,
        logger=platform.logger,
        detector_backend="mock",  # Using mock for demo (no dependencies)
        detector_model="yolov8n",
        detector_confidence=0.5,
        feature_backend="mock",  # Using mock for demo
        feature_model="ViT-B/32",
        device="cpu",
        track_objects=True,
        compute_relations=True,
    )
    
    stats = pipeline.get_statistics()
    print(f"   ✓ Pipeline created")
    print(f"     - Detector: {stats['detector']['backend']} ({stats['detector']['model']})")
    print(f"     - Feature extractor: {stats['feature_extractor']['backend']} (dim={stats['feature_extractor']['feature_dim']})")
    print(f"     - Confidence threshold: {stats['detector']['confidence_threshold']}")
    
    # Demo 1: Process single image
    print("\n" + "=" * 70)
    print("DEMO 1: Processing Single Image")
    print("=" * 70)
    
    print("\n   Creating mock RGB image (480x640)...")
    image = create_mock_image()
    print(f"   ✓ Image created: shape={image.shape}, dtype={image.dtype}")
    
    print("\n   Processing image through pipeline...")
    camera_pose = np.array([0.0, 0.0, 1.5], dtype=np.float32)  # Camera 1.5m high
    result = pipeline.process_image(image, camera_pose=camera_pose)
    
    print(f"\n   ✓ Processing complete!")
    print(f"     - Objects detected: {len(result.detections)}")
    print(f"     - Objects added to scene graph: {len(result.objects_added)}")
    print(f"     - Embeddings stored in UMA: {len(result.embeddings_stored)}")
    print(f"     - Spatial relations computed: {result.spatial_relations}")
    print(f"     - Processing time: {result.processing_time * 1000:.2f}ms")
    
    # Show detections
    print("\n   Detected objects:")
    for i, det in enumerate(result.detections):
        print(f"     {i+1}. {det.class_name}")
        print(f"        - Confidence: {det.confidence:.2f}")
        print(f"        - Bbox: ({det.bbox[0]:.0f}, {det.bbox[1]:.0f}) to ({det.bbox[2]:.0f}, {det.bbox[3]:.0f})")
        print(f"        - Center: ({det.get_bbox_center()[0]:.0f}, {det.get_bbox_center()[1]:.0f})")
        print(f"        - Area: {det.get_bbox_area():.0f}px²")
    
    # Demo 2: Scene Graph Inspection
    print("\n" + "=" * 70)
    print("DEMO 2: Scene Graph After Perception")
    print("=" * 70)
    
    sg_stats = scene_graph.get_statistics()
    print(f"\n   Scene Graph Statistics:")
    print(f"   - Total nodes: {sg_stats['total_nodes']}")
    print(f"   - Total edges: {sg_stats['total_edges']}")
    print(f"   - Nodes by layer: {sg_stats['nodes_by_layer']}")
    print(f"   - Edges by relation: {sg_stats['edges_by_relation']}")
    
    # Get objects
    obj_layer = ObjectLayer(scene_graph)
    objects = obj_layer.get_nodes()
    
    print(f"\n   Objects in scene graph (L1):")
    for obj in objects:
        print(f"   - {obj.attributes.get('class', 'unknown')} (ID: {obj.id[:8]}...)")
        print(f"     Position: {obj.position}")
        print(f"     Confidence: {obj.confidence:.2f}")
        if obj.embedding_key:
            print(f"     Embedding: {obj.embedding_key}")
            # Check UMA
            if platform.uma.exists(obj.embedding_key):
                emb_buffer = platform.uma.get(obj.embedding_key)
                print(f"       - Stored in UMA: shape={emb_buffer.shape}, device={emb_buffer.device}")
    
    # Demo 3: Process multiple frames (video simulation)
    print("\n" + "=" * 70)
    print("DEMO 3: Video Stream Simulation (3 frames)")
    print("=" * 70)
    
    # Clear scene graph for fresh start
    scene_graph.clear(layer=LayerType.L1)
    pipeline.clear_tracking()
    print("\n   Cleared scene graph for video demo")
    
    for frame_id in range(3):
        print(f"\n   Processing frame {frame_id + 1}/3...")
        
        # Create slightly different image for each frame
        image = create_mock_image()
        # Add some variation
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Process frame
        result = pipeline.process_video_frame(
            frame=image,
            frame_id=frame_id,
            camera_pose=camera_pose,
        )
        
        print(f"     ✓ Frame {frame_id + 1}: {len(result.objects_added)} objects, "
              f"{result.processing_time * 1000:.2f}ms")
    
    # Check tracked objects
    tracked = pipeline.get_tracked_objects()
    print(f"\n   ✓ Video processing complete")
    print(f"   ✓ Tracked objects: {len(tracked)}")
    for class_name, node in tracked.items():
        print(f"     - {class_name}: last seen at {node.position}")
    
    # Final scene graph stats
    final_stats = scene_graph.get_statistics()
    print(f"\n   Final scene graph:")
    print(f"   - Total objects: {final_stats['total_nodes']}")
    print(f"   - Total relations: {final_stats['total_edges']}")
    
    # Demo 4: Feature Similarity
    print("\n" + "=" * 70)
    print("DEMO 4: Visual Feature Analysis")
    print("=" * 70)
    
    objects = obj_layer.get_nodes()
    if len(objects) >= 2:
        print(f"\n   Analyzing visual features of {len(objects)} objects...")
        
        # Get embeddings from UMA
        embeddings = []
        for obj in objects:
            if obj.embedding_key and platform.uma.exists(obj.embedding_key):
                emb_buffer = platform.uma.get(obj.embedding_key)
                embeddings.append((obj, emb_buffer.data))
        
        print(f"   ✓ Retrieved {len(embeddings)} embeddings from UMA")
        
        # Compute pairwise similarities
        if len(embeddings) >= 2:
            from nesy.perception import FeatureVector
            
            print(f"\n   Pairwise cosine similarities:")
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    obj1, emb1 = embeddings[i]
                    obj2, emb2 = embeddings[j]
                    
                    # Create FeatureVector objects
                    feat1 = FeatureVector(emb1, len(emb1), "mock", normalized=True)
                    feat2 = FeatureVector(emb2, len(emb2), "mock", normalized=True)
                    
                    # Compute similarity
                    sim = feat1.cosine_similarity(feat2)
                    
                    print(f"     - {obj1.attributes.get('class')} ↔ {obj2.attributes.get('class')}: {sim:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PERCEPTION PIPELINE DEMO COMPLETE!")
    print("=" * 70)
    
    print("\n   Key Takeaways:")
    print("   ✓ Object detection from images (mock backend for demo)")
    print("   ✓ Visual feature extraction (512-dim embeddings)")
    print("   ✓ Automatic scene graph updates (L1 objects)")
    print("   ✓ Zero-copy embedding storage in UMA")
    print("   ✓ Spatial relation computation")
    print("   ✓ Object tracking across frames")
    print("   ✓ Feature similarity analysis")
    
    print("\n   Production Setup:")
    print("   - Replace 'mock' with 'yolo' for real object detection")
    print("   - Replace 'mock' with 'clip' for CLIP visual embeddings")
    print("   - Add RGB-D camera for depth estimation")
    print("   - Use SLAM for camera pose tracking")
    
    print("\n   Next Steps:")
    print("   - Week 6: VSA grounding (neural-symbolic binding)")
    print("   - Week 7: Reasoning over perceived scene graph")
    print("   - Integration with robot manipulation")


if __name__ == "__main__":
    main()
