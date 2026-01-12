"""
Kaggle Script: Extract Cycle Subgraphs from Pre-processed Graph Data

This script processes the graph data exported from local_verilog_to_graphs.py.
It extracts ALL cycles (SCCs) combined into a single subgraph per design and labels
based on whether any gate in the cycles is marked as trojan in result{N}.txt files.

Upload the following files to a Kaggle dataset:
    - all_graphs.json
    - all_graphs_metadata.json
    - result{0-19}.txt files (for trojan gate labels)

Then run this notebook/script on Kaggle.
"""

# =============================================================================
# CELL 1: Install dependencies (run this first on Kaggle)
# =============================================================================
# !pip install networkx torch torch-geometric

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import gc
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import networkx as nx

# =============================================================================
# CONFIGURATION - Adjust these paths for Kaggle
# =============================================================================
# For Kaggle, use:
# INPUT_DIR = "/kaggle/input/your-dataset-name"
# OUTPUT_DIR = "/kaggle/working"
# TROJAN_RESULT_DIR = "/kaggle/input/your-dataset-name"

# For local testing (relative to script location):
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
TROJAN_RESULT_DIR = os.path.join(SCRIPT_DIR, "release_all(20250728)", "trojan")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CELL 3: Trojan Gate Loading
# =============================================================================

def load_trojan_gates_from_result_files() -> Dict[int, Set[str]]:
    """
    Load trojan gates from result{N}.txt files.
    
    Returns:
        Dict mapping design_id -> set of trojan gate names
    """
    trojan_gates_by_design = {}
    
    # Designs 0-19 are trojaned, 20-29 are clean
    for design_id in range(20):
        result_file = os.path.join(TROJAN_RESULT_DIR, f"result{design_id}.txt")
        trojan_gates = set()
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    content = f.read()
                
                # Parse between TROJAN_GATES and END_TROJAN_GATES
                in_gates_section = False
                for line in content.split('\n'):
                    line = line.strip()
                    if line == 'TROJAN_GATES':
                        in_gates_section = True
                        continue
                    elif line == 'END_TROJAN_GATES':
                        in_gates_section = False
                        continue
                    elif in_gates_section and line:
                        trojan_gates.add(line)
                
                print(f"  Design {design_id}: {len(trojan_gates)} trojan gates loaded")
            except Exception as e:
                print(f"  Warning: Could not read {result_file}: {e}")
        else:
            print(f"  Warning: Result file not found: {result_file}")
        
        trojan_gates_by_design[design_id] = trojan_gates
    
    # Designs 20-29 are clean (no trojan gates)
    for design_id in range(20, 30):
        trojan_gates_by_design[design_id] = set()
    
    return trojan_gates_by_design


# =============================================================================
# CELL 4: Data Classes
# =============================================================================

@dataclass
class ReconstructedGraph:
    """A graph reconstructed from the JSON data."""
    design_id: int
    module_name: str
    is_trojaned: bool
    trojan_gates: Set[str]
    num_nodes: int
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    graph_features: np.ndarray
    edges: List[Tuple[int, int]]
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]
    node_types: Dict[str, str]
    depth: np.ndarray
    fan_in: np.ndarray
    fan_out: np.ndarray
    fan_in_norm: np.ndarray
    fan_out_norm: np.ndarray
    depth_norm: np.ndarray
    in_cycle: np.ndarray


@dataclass  
class CycleSubgraph:
    """Represents a subgraph containing ALL cycle/feedback regions from a design."""
    design_id: int
    original_module_name: str
    is_design_trojaned: bool
    num_nodes: int
    num_edges: int
    num_cycles: int  # Number of SCCs found
    cycle_nodes: Set[int]
    extended_nodes: Set[int]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    graph_features: np.ndarray
    node_mapping: Dict[int, int]
    reverse_mapping: Dict[int, int]
    node_types: Dict[int, str]
    gate_names_in_cycle: Set[str]
    gate_names_in_subgraph: Set[str]
    trojan_gates_in_cycle: Set[str]
    trojan_gates_in_subgraph: Set[str]
    label: int  # 0 = safe, 1 = contains trojan gates in subgraph
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'design_id': self.design_id,
            'original_module_name': self.original_module_name,
            'is_design_trojaned': self.is_design_trojaned,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_cycles': self.num_cycles,
            'num_cycle_nodes': len(self.cycle_nodes),
            'num_extended_nodes': len(self.extended_nodes),
            'node_features': self.node_features.tolist(),
            'edge_index': self.edge_index.tolist(),
            'edge_features': self.edge_features.tolist() if self.edge_features.size > 0 else [],
            'graph_features': self.graph_features.tolist(),
            'node_types': {str(k): v for k, v in self.node_types.items()},
            'gate_names_in_cycle': list(self.gate_names_in_cycle),
            'num_gates_in_cycle': len(self.gate_names_in_cycle),
            'gate_names_in_subgraph': list(self.gate_names_in_subgraph),
            'num_gates_in_subgraph': len(self.gate_names_in_subgraph),
            'trojan_gates_in_cycle': list(self.trojan_gates_in_cycle),
            'num_trojan_gates_in_cycle': len(self.trojan_gates_in_cycle),
            'trojan_gates_in_subgraph': list(self.trojan_gates_in_subgraph),
            'num_trojan_gates_in_subgraph': len(self.trojan_gates_in_subgraph),
            'label': self.label
        }


# =============================================================================
# CELL 5: Load Graph Data (Streaming)
# =============================================================================

def load_graphs_streaming(json_path: str, trojan_gates_by_design: Dict[int, Set[str]]):
    """Generator to load graphs one at a time to save memory.
    
    Args:
        json_path: Path to the all_graphs.json file
        trojan_gates_by_design: Dict mapping design_id -> set of trojan gate names
    """
    print(f"Loading graphs from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} graphs")
    
    for g in data:
        design_id = g['design_id']
        # Override trojan_gates with data from result files
        trojan_gates = trojan_gates_by_design.get(design_id, set())
        
        graph = ReconstructedGraph(
            design_id=design_id,
            module_name=g['module_name'],
            is_trojaned=g['is_trojaned'],
            trojan_gates=trojan_gates,  # Use result file data
            num_nodes=g['num_nodes'],
            node_features=np.array(g['node_features'], dtype=np.float32),
            edge_index=np.array(g['edge_index'], dtype=np.int64),
            edge_features=np.array(g['edge_features'], dtype=np.float32),
            graph_features=np.array(g['graph_features'], dtype=np.float32),
            edges=[(e[0], e[1]) for e in g['edges']],
            node_to_idx=g['node_to_idx'],
            idx_to_node={int(k): v for k, v in g['idx_to_node'].items()},
            node_types=g['node_types'],
            depth=np.array(g['depth'], dtype=np.float32),
            fan_in=np.array(g['fan_in'], dtype=np.float32),
            fan_out=np.array(g['fan_out'], dtype=np.float32),
            fan_in_norm=np.array(g['fan_in_norm'], dtype=np.float32),
            fan_out_norm=np.array(g['fan_out_norm'], dtype=np.float32),
            depth_norm=np.array(g['depth_norm'], dtype=np.float32),
            in_cycle=np.array(g['in_cycle'], dtype=np.float32),
        )
        yield graph
        del graph
    
    del data
    gc.collect()


# =============================================================================
# CELL 6: Cycle Extraction
# =============================================================================

# Gate type indices for neighborhood analysis
GATE_TYPE_INDICES = {
    'input': 0, 'output': 1, 'wire': 2, 'and': 3, 'nand': 4,
    'or': 5, 'nor': 6, 'xor': 7, 'xnor': 8, 'not': 9,
    'buf': 10, 'dff': 11, 'mux': 12, 'unknown': 13
}
NUM_GATE_TYPES = len(GATE_TYPE_INDICES)
INVERTING_GATES = {'not', 'nand', 'nor', 'xnor'}
SEQUENTIAL_GATES = {'dff'}


def compute_neighborhood_features(G: nx.DiGraph, 
                                   node_types: Dict[str, str],
                                   idx_to_node: Dict[int, str],
                                   fan_in: np.ndarray,
                                   fan_out: np.ndarray,
                                   k_hops: int = 2) -> np.ndarray:
    """
    Compute local neighborhood features for each node.
    
    Features per node (total = 7):
    - avg_neighbor_fanin: Average fanin of k-hop neighbors
    - avg_neighbor_fanout: Average fanout of k-hop neighbors  
    - inverter_density: Fraction of inverting gates in k-hop neighborhood
    - sequential_density: Fraction of sequential elements (DFFs) in k-hop neighborhood
    - rare_gate_density: Fraction of rare gates (xor, xnor, mux) in neighborhood
    - neighbor_count: Number of nodes in k-hop neighborhood (normalized)
    - self_loop_indicator: Whether node is part of a self-feedback path
    """
    num_nodes = G.number_of_nodes()
    
    # Initialize feature arrays
    avg_neighbor_fanin = np.zeros(num_nodes, dtype=np.float32)
    avg_neighbor_fanout = np.zeros(num_nodes, dtype=np.float32)
    inverter_density = np.zeros(num_nodes, dtype=np.float32)
    sequential_density = np.zeros(num_nodes, dtype=np.float32)
    rare_gate_density = np.zeros(num_nodes, dtype=np.float32)
    neighbor_count = np.zeros(num_nodes, dtype=np.float32)
    self_loop_indicator = np.zeros(num_nodes, dtype=np.float32)
    
    # Rare gates that might indicate trojan logic
    rare_gates = {'xor', 'xnor', 'mux'}
    
    for node in range(num_nodes):
        # Get k-hop neighborhood
        neighbors = set()
        current_frontier = {node}
        
        for _ in range(k_hops):
            next_frontier = set()
            for n in current_frontier:
                next_frontier.update(G.predecessors(n))
                next_frontier.update(G.successors(n))
            neighbors.update(next_frontier)
            current_frontier = next_frontier - neighbors
        
        neighbors.discard(node)  # Remove self
        
        if len(neighbors) == 0:
            continue
        
        neighbor_count[node] = len(neighbors) / 100.0  # Normalize
        
        # Compute statistics over neighbors
        neighbor_fanins = [fan_in[n] for n in neighbors]
        neighbor_fanouts = [fan_out[n] for n in neighbors]
        
        avg_neighbor_fanin[node] = np.mean(neighbor_fanins) / 10.0  # Normalize
        avg_neighbor_fanout[node] = np.mean(neighbor_fanouts) / 10.0
        
        # Count gate types in neighborhood
        inverting_count = 0
        sequential_count = 0
        rare_count = 0
        
        for n in neighbors:
            node_name = idx_to_node.get(n, '')
            gate_type = node_types.get(node_name, 'unknown').lower()
            
            if gate_type in INVERTING_GATES:
                inverting_count += 1
            if gate_type in SEQUENTIAL_GATES:
                sequential_count += 1
            if gate_type in rare_gates:
                rare_count += 1
        
        inverter_density[node] = inverting_count / len(neighbors)
        sequential_density[node] = sequential_count / len(neighbors)
        rare_gate_density[node] = rare_count / len(neighbors)
        
        # Check for self-feedback (path from node back to itself)
        try:
            if nx.has_path(G, node, node):
                self_loop_indicator[node] = 1.0
        except:
            pass
    
    # Stack all features
    features = np.column_stack([
        avg_neighbor_fanin,
        avg_neighbor_fanout,
        inverter_density,
        sequential_density,
        rare_gate_density,
        neighbor_count,
        self_loop_indicator
    ])
    
    return features.astype(np.float32)


class CycleExtractor:
    """Extracts ALL cycles (SCCs) combined into a single subgraph per design."""
    
    def __init__(self, 
                 include_neighbors: int = 1,
                 min_cycle_nodes: int = 2,
                 include_feedback_edges: bool = True):
        self.include_neighbors = include_neighbors
        self.min_cycle_nodes = min_cycle_nodes
        self.include_feedback_edges = include_feedback_edges
    
    def extract_from_graph(self, graph: ReconstructedGraph) -> Optional[CycleSubgraph]:
        """
        Extract ALL cycles (SCCs) combined into a single subgraph.
        
        Returns:
            A single CycleSubgraph containing all cycle nodes, or None if no cycles
        """
        num_nodes = graph.num_nodes
        if num_nodes == 0:
            return None
        
        # Build NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(graph.edges)
        
        # Find all SCCs (Strongly Connected Components) - O(V+E) algorithm
        # Only keep SCCs with size >= min_cycle_nodes (actual cycles)
        sccs = [scc for scc in nx.strongly_connected_components(G) 
                if len(scc) >= self.min_cycle_nodes]
        
        # Combine all SCC nodes
        cycle_nodes = set()
        for scc in sccs:
            cycle_nodes.update(scc)
        
        # Also find feedback edges (edges where source depth > dest depth)
        feedback_nodes = set()
        if self.include_feedback_edges:
            for src, dst in graph.edges:
                if graph.depth[src] > graph.depth[dst]:
                    feedback_nodes.add(src)
                    feedback_nodes.add(dst)
        
        # Combine cycle nodes and feedback nodes
        core_nodes = cycle_nodes | feedback_nodes
        
        if len(core_nodes) < self.min_cycle_nodes:
            return None
        
        # Extend to include neighbors
        extended_nodes = set(core_nodes)
        if self.include_neighbors > 0:
            current_frontier = core_nodes
            for _ in range(self.include_neighbors):
                new_frontier = set()
                for node in current_frontier:
                    new_frontier.update(G.predecessors(node))
                    new_frontier.update(G.successors(node))
                extended_nodes.update(new_frontier)
                current_frontier = new_frontier - extended_nodes
        
        # Pre-compute neighborhood features for the full graph
        neighborhood_features_full = compute_neighborhood_features(
            G=G,
            node_types=graph.node_types,
            idx_to_node=graph.idx_to_node,
            fan_in=graph.fan_in,
            fan_out=graph.fan_out,
            k_hops=2
        )
        
        # Create subgraph
        return self._create_subgraph(
            graph, G, cycle_nodes, extended_nodes, 
            neighborhood_features_full, len(sccs)
        )
    
    def _create_subgraph(self,
                         graph: ReconstructedGraph,
                         G: nx.DiGraph,
                         cycle_nodes: Set[int],
                         extended_nodes: Set[int],
                         neighborhood_features_full: np.ndarray,
                         num_cycles: int) -> CycleSubgraph:
        """Create a CycleSubgraph from all SCCs combined."""
        
        # Create mapping from old indices to new indices
        sorted_nodes = sorted(extended_nodes)
        node_mapping = {old: new for new, old in enumerate(sorted_nodes)}
        reverse_mapping = {new: old for old, new in node_mapping.items()}
        
        # Extract edges within the subgraph
        new_edges = []
        for src, dst in graph.edges:
            if src in extended_nodes and dst in extended_nodes:
                new_edges.append((node_mapping[src], node_mapping[dst]))
        
        # Get node features for the subgraph
        subgraph_features = graph.node_features[sorted_nodes]
        
        # Add cycle membership as additional feature
        cycle_membership = np.zeros((len(sorted_nodes), 1), dtype=np.float32)
        for new_idx, old_idx in reverse_mapping.items():
            if old_idx in cycle_nodes:
                cycle_membership[new_idx] = 1.0
        
        # Extract neighborhood features for subgraph nodes
        neighborhood_features = neighborhood_features_full[sorted_nodes]
        
        # Combine all node features: original (18) + cycle_membership (1) + neighborhood (7) = 26 features
        node_features = np.hstack([subgraph_features, cycle_membership, neighborhood_features])
        
        # Create edge features
        if new_edges:
            edge_features = []
            for new_src, new_dst in new_edges:
                old_src = reverse_mapping[new_src]
                old_dst = reverse_mapping[new_dst]
                fan_out_src = graph.fan_out_norm[old_src]
                fan_in_dst = graph.fan_in_norm[old_dst]
                is_feedback = 1.0 if graph.depth[old_src] > graph.depth[old_dst] else 0.0
                is_in_cycle = 1.0 if (old_src in cycle_nodes and old_dst in cycle_nodes) else 0.0
                edge_features.append([fan_out_src, fan_in_dst, is_feedback, is_in_cycle])
            edge_features = np.array(edge_features, dtype=np.float32)
            edge_index = np.array(new_edges, dtype=np.int64).T
        else:
            edge_features = np.zeros((0, 4), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        # Compute graph-level features for the subgraph
        graph_features = self._compute_subgraph_features(
            G.subgraph(extended_nodes), cycle_nodes, extended_nodes, num_cycles
        )
        
        # Get node types and gate names IN THE SUBGRAPH (cycles + neighbors)
        node_types = {}
        gate_names_in_subgraph = set()
        gate_names_in_cycle = set()
        for new_idx, old_idx in reverse_mapping.items():
            old_node_name = graph.idx_to_node[old_idx]
            node_types[new_idx] = graph.node_types.get(old_node_name, 'unknown')
            # Skip PI_ and PO_ prefixes
            if not old_node_name.startswith('PI_') and not old_node_name.startswith('PO_'):
                gate_names_in_subgraph.add(old_node_name)
                # Track gates that are in the cycle core (not just neighbors)
                if old_idx in cycle_nodes:
                    gate_names_in_cycle.add(old_node_name)
        
        # Check which trojan gates are in the subgraph (cycles + surrounding)
        trojan_gates_in_subgraph = graph.trojan_gates & gate_names_in_subgraph
        trojan_gates_in_cycle = graph.trojan_gates & gate_names_in_cycle
        
        # Label: 1 if any trojan gate is in the subgraph (including neighbors), else 0
        label = 1 if len(trojan_gates_in_subgraph) > 0 else 0
        
        return CycleSubgraph(
            design_id=graph.design_id,
            original_module_name=graph.module_name,
            is_design_trojaned=graph.is_trojaned,
            num_nodes=len(extended_nodes),
            num_edges=len(new_edges),
            num_cycles=num_cycles,
            cycle_nodes=cycle_nodes,
            extended_nodes=extended_nodes,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            graph_features=graph_features,
            node_mapping=node_mapping,
            reverse_mapping=reverse_mapping,
            node_types=node_types,
            gate_names_in_cycle=gate_names_in_cycle,
            gate_names_in_subgraph=gate_names_in_subgraph,
            trojan_gates_in_cycle=trojan_gates_in_cycle,
            trojan_gates_in_subgraph=trojan_gates_in_subgraph,
            label=label
        )
    
    def _compute_subgraph_features(self,
                                   subgraph: nx.DiGraph,
                                   cycle_nodes: Set[int],
                                   extended_nodes: Set[int],
                                   num_cycles: int) -> np.ndarray:
        """Compute graph-level features for the cycle subgraph."""
        num_nodes = len(extended_nodes)
        num_edges = subgraph.number_of_edges()
        
        cycle_node_fraction = len(cycle_nodes) / max(num_nodes, 1)
        
        # SCCs in subgraph
        try:
            sccs = list(nx.strongly_connected_components(subgraph))
            num_sccs = len(sccs)
            largest_scc = max(len(scc) for scc in sccs) if sccs else 0
        except:
            num_sccs = 0
            largest_scc = 0
        
        density = nx.density(subgraph) if num_nodes > 1 else 0
        
        in_degrees = [d for _, d in subgraph.in_degree()]
        out_degrees = [d for _, d in subgraph.out_degree()]
        avg_in_degree = np.mean(in_degrees) if in_degrees else 0
        avg_out_degree = np.mean(out_degrees) if out_degrees else 0
        
        features = np.array([
            num_nodes / 100,
            num_edges / 200,
            num_cycles / 10,
            cycle_node_fraction,
            num_sccs / 10,
            largest_scc / 20,
            density,
            avg_in_degree / 5,
            avg_out_degree / 5,
        ], dtype=np.float32)
        
        return features


# =============================================================================
# CELL 7: Save Functions
# =============================================================================

def save_cycle_subgraphs_json(subgraphs: List[CycleSubgraph], output_path: str):
    """Save cycle subgraphs to JSON format."""
    data = [sg.to_dict() for sg in subgraphs]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(subgraphs)} cycle subgraphs to {output_path}")


def save_cycle_subgraphs_npz(subgraphs: List[CycleSubgraph], output_path: str):
    """Save cycle subgraphs in NumPy compressed format."""
    data = {'num_graphs': len(subgraphs)}
    
    for i, sg in enumerate(subgraphs):
        data[f'node_features_{i}'] = sg.node_features
        data[f'edge_index_{i}'] = sg.edge_index
        data[f'edge_features_{i}'] = sg.edge_features
        data[f'graph_features_{i}'] = sg.graph_features
        data[f'label_{i}'] = sg.label
        data[f'design_id_{i}'] = sg.design_id
        data[f'num_nodes_{i}'] = sg.num_nodes
        data[f'num_edges_{i}'] = sg.num_edges
        data[f'num_cycles_{i}'] = sg.num_cycles
    
    np.savez_compressed(output_path, **data)
    print(f"Saved {len(subgraphs)} cycle subgraphs to {output_path}")


def save_cycle_subgraphs_pytorch(subgraphs: List[CycleSubgraph], output_path: str):
    """Save cycle subgraphs in PyTorch Geometric format."""
    try:
        import torch
        from torch_geometric.data import Data
        
        data_list = []
        for sg in subgraphs:
            x = torch.tensor(sg.node_features, dtype=torch.float)
            edge_index = torch.tensor(sg.edge_index, dtype=torch.long)
            edge_attr = torch.tensor(sg.edge_features, dtype=torch.float) if sg.edge_features.size > 0 else None
            graph_attr = torch.tensor(sg.graph_features, dtype=torch.float)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                graph_attr=graph_attr,
                y=torch.tensor([sg.label], dtype=torch.long)
            )
            data.design_id = sg.design_id
            data.num_cycles = sg.num_cycles
            data_list.append(data)
        
        torch.save(data_list, output_path)
        print(f"Saved {len(data_list)} cycle subgraphs to {output_path}")
        
    except ImportError:
        print("PyTorch Geometric not installed. Skipping .pt format.")


# =============================================================================
# CELL 8: Main Execution
# =============================================================================

def main():
    """Main function to extract ALL cycles combined into one subgraph per design."""
    
    print("="*60)
    print("CYCLE EXTRACTOR - Combined SCC Extraction")
    print("="*60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Trojan labels: {TROJAN_RESULT_DIR}")
    
    # Load trojan gates from result files
    print("\nLoading trojan gate labels from result files...")
    trojan_gates_by_design = load_trojan_gates_from_result_files()
    total_trojan_gates = sum(len(gates) for gates in trojan_gates_by_design.values())
    print(f"Total trojan gates loaded: {total_trojan_gates}")
    
    # Create extractor
    extractor = CycleExtractor(
        include_neighbors=2,  # Larger neighborhood for more context
        min_cycle_nodes=2,
        include_feedback_edges=True
    )
    
    # Extract cycles (streaming to save memory)
    print("\nExtracting all cycles combined into one subgraph per design...")
    json_path = os.path.join(INPUT_DIR, "all_graphs.json")
    
    all_subgraphs = []
    
    for graph in load_graphs_streaming(json_path, trojan_gates_by_design):
        subgraph = extractor.extract_from_graph(graph)
        if subgraph is not None:
            all_subgraphs.append(subgraph)
            print(f"  design{graph.design_id}: {subgraph.num_nodes} nodes, "
                  f"{subgraph.num_cycles} SCCs, "
                  f"{len(subgraph.trojan_gates_in_cycle)} trojan in cycles, "
                  f"{len(subgraph.trojan_gates_in_subgraph)} trojan in subgraph, "
                  f"label={subgraph.label}")
        else:
            print(f"  design{graph.design_id}: No cycles found")
        gc.collect()
    
    if not all_subgraphs:
        print("\nNo cycle subgraphs extracted!")
        return
    
    # Print summary stats
    labels = [sg.label for sg in all_subgraphs]
    num_trojaned = sum(labels)
    num_clean = len(labels) - num_trojaned
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total designs with cycles: {len(all_subgraphs)}")
    print(f"  - Trojaned (label=1): {num_trojaned}")
    print(f"  - Clean (label=0): {num_clean}")
    print(f"Class balance: {num_trojaned}/{len(all_subgraphs)} = {100*num_trojaned/len(all_subgraphs):.1f}% trojaned")
    
    # Save outputs
    print("\nSaving...")
    output_prefix = os.path.join(OUTPUT_DIR, "feedback_dataset")
    
    save_cycle_subgraphs_json(all_subgraphs, f"{output_prefix}.json")
    save_cycle_subgraphs_npz(all_subgraphs, f"{output_prefix}.npz")
    save_cycle_subgraphs_pytorch(all_subgraphs, f"{output_prefix}.pt")
    
    # Save extraction metadata
    metadata = {
        'total_designs': len(all_subgraphs),
        'trojaned_designs': num_trojaned,
        'clean_designs': num_clean,
        'class_balance': num_trojaned / len(all_subgraphs),
        'extraction_params': {
            'include_neighbors': extractor.include_neighbors,
            'min_cycle_nodes': extractor.min_cycle_nodes,
            'include_feedback_edges': extractor.include_feedback_edges
        }
    }
    with open(os.path.join(OUTPUT_DIR, "extraction_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved extraction metadata to {os.path.join(OUTPUT_DIR, 'extraction_metadata.json')}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
