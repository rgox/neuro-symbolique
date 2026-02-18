import argparse
import sys
import logging
from nesy.pipeline import NeuralSymbolicPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="NeSy Visualization Tool")
    parser.add_argument("--scene-graph", action="store_true", help="Visualize scene graph")
    parser.add_argument("--pipeline", action="store_true", help="Visualize full pipeline")
    args = parser.parse_args()

    if args.scene_graph:
        print("Initializing Scene Graph Visualization...")
        try:
            # Initialize pipeline
            pipeline = NeuralSymbolicPipeline()
            sg = pipeline.scene_graph
            
            # Create some dummy data if empty
            if not sg.nodes:
                print("Empty scene graph. Adding demo nodes...")
                from nesy.world_model.scene_graph import Node, NodeType, LayerType, RelationType
                
                sg.add_node(
                    layer=LayerType.L1,
                    node_type=NodeType.OBJECT,
                    position=[1.0, 2.0, 0.5],
                    attributes={"class": "cup", "color": "red"},
                    node_id="cup_1"
                )
                sg.add_node(
                    layer=LayerType.L1,
                    node_type=NodeType.OBJECT,
                    position=[1.0, 2.0, 0.0],
                    attributes={"class": "table"},
                    node_id="table_1"
                )
                sg.add_edge("cup_1", "table_1", RelationType.ON)
            
            print(f"Scene Graph State:")
            print(f"  Nodes: {len(sg.nodes)}")
            print(f"  Edges: {len(sg.edges)}")
            
            for node_id, node in sg.nodes.items():
                print(f"  - {node_id}: {node.attributes}")
                
            print("Visualization complete.")
            
        except Exception as e:
            logger.error(f"Error visualizing scene graph: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif args.pipeline:
        print("Pipeline visualization not implemented yet.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
