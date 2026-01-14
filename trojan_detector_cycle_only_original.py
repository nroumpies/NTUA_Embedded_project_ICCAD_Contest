#!/usr/bin/env python3
"""
Hardware Trojan Detector (Cycle-Only Model)

Uses only the cycle-level extraction model for trojan detection.
This model focuses on feedback loops and cycle structures in the circuit.

Usage:
    python trojan_detector_cycle_only.py -netlist <path/to/design.v> -output <path/to/result.txt>
"""

import os
import sys
import re
import argparse
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    #"cycle_RandomForest_Deep.pkl"
    "cycle_AdaBoost.pkl"
)

# Gate type encoding (must match training)
GATE_TYPES = {
    'input': 0, 'output': 1, 'wire': 2, 'and': 3, 'nand': 4,
    'or': 5, 'nor': 6, 'xor': 7, 'xnor': 8, 'not': 9,
    'buf': 10, 'dff': 11, 'mux': 12, 'unknown': 13
}
NUM_GATE_TYPES = len(GATE_TYPES)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Gate:
    """Represents a gate in the circuit."""
    name: str
    gate_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class VerilogModule:
    """Represents a parsed Verilog module."""
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    wires: List[str] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)


@dataclass
class CycleSubgraph:
    """Represents an extracted cycle/feedback subgraph."""
    num_nodes: int
    num_edges: int
    num_cycles: int
    node_features: np.ndarray
    edge_index: List[List[int]]
    cycle_nodes: Set[int]
    gate_names: Set[str]


# =============================================================================
# Verilog Parser
# =============================================================================

class VerilogParser:
    """Parser for structural Verilog netlists."""
    
    def __init__(self):
        self.modules: List[VerilogModule] = []
        
    def parse_file(self, filepath: str) -> List[VerilogModule]:
        """Parse a Verilog file and extract all modules."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> List[VerilogModule]:
        """Parse Verilog content string."""
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Find all modules
        module_pattern = r'module\s+(\w+)\s*\(([^)]*)\)\s*;(.*?)endmodule'
        modules = re.findall(module_pattern, content, re.DOTALL)
        
        self.modules = []
        for module_name, ports, body in modules:
            module = self._parse_module(module_name, ports, body)
            self.modules.append(module)
            
        return self.modules
    
    def _parse_module(self, name: str, ports: str, body: str) -> VerilogModule:
        """Parse a single module."""
        module = VerilogModule(name=name)
        
        lines = body.split(';')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('input'):
                module.inputs.extend(self._parse_declaration(line, 'input'))
            elif line.startswith('output'):
                module.outputs.extend(self._parse_declaration(line, 'output'))
            elif line.startswith('wire'):
                module.wires.extend(self._parse_declaration(line, 'wire'))
            else:
                gate = self._parse_gate(line)
                if gate:
                    module.gates.append(gate)
                    
        return module
    
    def _parse_declaration(self, line: str, decl_type: str) -> List[str]:
        """Parse input/output/wire declarations."""
        signals = []
        line = re.sub(rf'^{decl_type}\s+', '', line)
        
        # Handle bus declarations
        bus_pattern = r'\[(\d+):(\d+)\]\s*(\w+)'
        bus_match = re.search(bus_pattern, line)
        if bus_match:
            high, low, name = bus_match.groups()
            for i in range(int(low), int(high) + 1):
                signals.append(f"{name}[{i}]")
            line = re.sub(bus_pattern, '', line)
        
        remaining = [s.strip() for s in line.split(',') if s.strip()]
        signals.extend(remaining)
        
        return signals
    
    def _parse_gate(self, line: str) -> Optional[Gate]:
        """Parse a gate instantiation."""
        line = line.strip()
        if not line:
            return None
            
        # DFF pattern
        dff_pattern = r'^dff\s+(\w+)\s*\((.+)\)'
        dff_match = re.match(dff_pattern, line, re.DOTALL)
        if dff_match:
            gate_name = dff_match.group(1)
            ports_str = dff_match.group(2)
            
            gate = Gate(name=gate_name, gate_type='dff')
            
            port_pattern = r'\.(\w+)\s*\(\s*([^)]+)\s*\)'
            for port_match in re.finditer(port_pattern, ports_str):
                port_name, signal = port_match.groups()
                signal = signal.strip()
                if port_name.lower() in ['d', 'clk', 'r']:
                    gate.inputs.append(signal)
                elif port_name.lower() == 'q':
                    gate.outputs.append(signal)
            
            return gate
        
        # Standard gate pattern
        std_pattern = r'^(\w+)\s+(\w+)\s*\(([^)]+)\)'
        std_match = re.match(std_pattern, line)
        if std_match:
            gate_type, gate_name, ports = std_match.groups()
            gate_type = gate_type.lower()
            
            if gate_type not in GATE_TYPES:
                gate_type = 'unknown'
            
            signals = [s.strip() for s in ports.split(',')]
            
            gate = Gate(name=gate_name, gate_type=gate_type)
            if signals:
                gate.outputs = [signals[0]]
                gate.inputs = signals[1:] if len(signals) > 1 else []
            
            return gate
            
        return None


# =============================================================================
# Circuit Graph Builder
# =============================================================================

class CircuitGraph:
    """Builds a directed graph from parsed Verilog."""
    
    def __init__(self, module: VerilogModule):
        self.module = module
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.node_types: Dict[str, str] = {}
        self.edges: List[Tuple[int, int]] = []
        self.signal_driver: Dict[str, str] = {}
        self.signal_consumers: Dict[str, List[str]] = defaultdict(list)
        
        self._build_graph()
        
    def _build_graph(self):
        """Build graph from module."""
        node_idx = 0
        
        # Primary inputs
        for inp in self.module.inputs:
            self.node_to_idx[f"PI_{inp}"] = node_idx
            self.idx_to_node[node_idx] = f"PI_{inp}"
            self.node_types[f"PI_{inp}"] = 'input'
            self.signal_driver[inp] = f"PI_{inp}"
            node_idx += 1
            
        # Primary outputs
        for out in self.module.outputs:
            self.node_to_idx[f"PO_{out}"] = node_idx
            self.idx_to_node[node_idx] = f"PO_{out}"
            self.node_types[f"PO_{out}"] = 'output'
            self.signal_consumers[out].append(f"PO_{out}")
            node_idx += 1
                
        # Gates
        for gate in self.module.gates:
            self.node_to_idx[gate.name] = node_idx
            self.idx_to_node[node_idx] = gate.name
            self.node_types[gate.name] = gate.gate_type
            
            for out_signal in gate.outputs:
                self.signal_driver[out_signal] = gate.name
            for in_signal in gate.inputs:
                self.signal_consumers[in_signal].append(gate.name)
            
            node_idx += 1
        
        # Create edges
        all_signals = set(self.signal_driver.keys()) | set(self.signal_consumers.keys())
        
        for signal in all_signals:
            if signal in self.signal_driver:
                driver = self.signal_driver[signal]
                driver_idx = self.node_to_idx.get(driver)
                
                if driver_idx is not None:
                    for consumer in self.signal_consumers.get(signal, []):
                        consumer_idx = self.node_to_idx.get(consumer)
                        if consumer_idx is not None:
                            self.edges.append((driver_idx, consumer_idx))
        
        self._compute_structural_features()
    
    def _compute_structural_features(self):
        """Compute fan-in, fan-out, depth, cycle membership."""
        num_nodes = len(self.node_to_idx)
        
        self.fan_in = np.zeros(num_nodes, dtype=np.float32)
        self.fan_out = np.zeros(num_nodes, dtype=np.float32)
        self.depth = np.zeros(num_nodes, dtype=np.float32)
        self.in_cycle = np.zeros(num_nodes, dtype=np.float32)
        
        for src, dst in self.edges:
            self.fan_out[src] += 1
            self.fan_in[dst] += 1
        
        # Normalize
        max_fan_in = max(self.fan_in.max(), 1)
        max_fan_out = max(self.fan_out.max(), 1)
        self.fan_in_norm = self.fan_in / max_fan_in
        self.fan_out_norm = self.fan_out / max_fan_out
        
        # Build NetworkX graph for cycle detection
        if num_nodes > 0:
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(self.edges)
            
            # Find nodes in cycles via SCCs
            try:
                sccs = list(nx.strongly_connected_components(G))
                for scc in sccs:
                    if len(scc) > 1:
                        for node in scc:
                            self.in_cycle[node] = 1.0
            except:
                pass
            
            # Compute depth (BFS from inputs)
            try:
                sources = [n for n in G.nodes() if G.in_degree(n) == 0]
                for src in sources:
                    for node, d in nx.single_source_shortest_path_length(G, src).items():
                        self.depth[node] = max(self.depth[node], d)
            except:
                pass
        
        max_depth = max(self.depth.max(), 1)
        self.depth_norm = self.depth / max_depth
    
    def get_node_features(self) -> np.ndarray:
        """Get node feature matrix (one-hot gate types + structural)."""
        num_nodes = len(self.node_to_idx)
        features = np.zeros((num_nodes, NUM_GATE_TYPES + 4), dtype=np.float32)
        
        for node, idx in self.node_to_idx.items():
            gate_type = self.node_types.get(node, 'unknown')
            type_idx = GATE_TYPES.get(gate_type, GATE_TYPES['unknown'])
            features[idx, type_idx] = 1.0
            
            features[idx, NUM_GATE_TYPES] = self.fan_in_norm[idx]
            features[idx, NUM_GATE_TYPES + 1] = self.fan_out_norm[idx]
            features[idx, NUM_GATE_TYPES + 2] = self.depth_norm[idx]
            features[idx, NUM_GATE_TYPES + 3] = self.in_cycle[idx]
            
        return features
    
    def get_edge_index(self) -> List[List[int]]:
        """Get edge index as [sources, targets]."""
        if not self.edges:
            return [[], []]
        edges = np.array(self.edges, dtype=np.int64).T
        return edges.tolist()


# =============================================================================
# Cycle Extraction
# =============================================================================

def extract_cycle_subgraph(graph: CircuitGraph, 
                           include_neighbors: int = 1,
                           max_cycles: int = 100) -> Optional[CycleSubgraph]:
    """Extract cycle/feedback subgraph from circuit."""
    num_nodes = len(graph.node_to_idx)
    if num_nodes == 0:
        return None
    
    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(graph.edges)
    
    # Find cycle nodes via SCCs
    cycle_nodes = set()
    cycle_count = 0
    
    try:
        sccs = list(nx.strongly_connected_components(G))
        for scc in sccs:
            if len(scc) > 1:
                cycle_nodes.update(scc)
                cycle_count += 1
        
        # Also try simple cycles (limited)
        if num_nodes < 1000:
            try:
                for i, cycle in enumerate(nx.simple_cycles(G)):
                    if i >= max_cycles:
                        break
                    cycle_nodes.update(cycle)
                    cycle_count += 1
            except:
                pass
    except:
        pass
    
    # Also find feedback edges
    feedback_nodes = set()
    try:
        for scc in sccs:
            if len(scc) > 1:
                feedback_nodes.update(scc)
    except:
        pass
    
    core_nodes = cycle_nodes | feedback_nodes
    
    if len(core_nodes) < 2:
        return None
    
    # Extend to include neighbors
    extended_nodes = set(core_nodes)
    if include_neighbors > 0:
        for _ in range(include_neighbors):
            new_nodes = set()
            for node in extended_nodes:
                new_nodes.update(G.predecessors(node))
                new_nodes.update(G.successors(node))
            extended_nodes.update(new_nodes)
    
    # Create subgraph
    sorted_nodes = sorted(extended_nodes)
    node_mapping = {old: new for new, old in enumerate(sorted_nodes)}
    
    # Extract edges
    new_edges_src = []
    new_edges_dst = []
    for src, dst in graph.edges:
        if src in extended_nodes and dst in extended_nodes:
            new_edges_src.append(node_mapping[src])
            new_edges_dst.append(node_mapping[dst])
    
    # Get node features for subgraph
    full_features = graph.get_node_features()
    subgraph_features = full_features[sorted_nodes]
    
    # Add structural features
    sub_fan_in = graph.fan_in_norm[sorted_nodes]
    sub_fan_out = graph.fan_out_norm[sorted_nodes]
    sub_depth = graph.depth_norm[sorted_nodes]
    sub_in_cycle_orig = graph.in_cycle[sorted_nodes]
    
    # Cycle membership in subgraph
    sub_in_cycle = np.zeros(len(sorted_nodes), dtype=np.float32)
    for new_idx, old_idx in enumerate(sorted_nodes):
        if old_idx in cycle_nodes:
            sub_in_cycle[new_idx] = 1.0
    
    # Combine features: gate types (14) + structural (4) + cycle membership (1)
    node_features = np.column_stack([
        subgraph_features,
        sub_fan_in.reshape(-1, 1),
        sub_fan_out.reshape(-1, 1),
        sub_depth.reshape(-1, 1),
        sub_in_cycle_orig.reshape(-1, 1),
        sub_in_cycle.reshape(-1, 1)
    ])
    
    # Get gate names
    gate_names = set()
    for old_idx in sorted_nodes:
        node_name = graph.idx_to_node.get(old_idx, '')
        if not node_name.startswith('PI_') and not node_name.startswith('PO_'):
            gate_names.add(node_name)
    
    return CycleSubgraph(
        num_nodes=len(sorted_nodes),
        num_edges=len(new_edges_src),
        num_cycles=cycle_count,
        node_features=node_features,
        edge_index=[new_edges_src, new_edges_dst],
        cycle_nodes=cycle_nodes,
        gate_names=gate_names
    )


# =============================================================================
# Feature Extraction (Cycle Features)
# =============================================================================

def compute_directed_features(G: nx.DiGraph, node_features: np.ndarray) -> Dict[str, float]:
    """Compute directed graph features."""
    features = {}
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        return {}
    
    # In-degree / Out-degree analysis
    in_degrees = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])
    
    features['source_node_fraction'] = np.sum(in_degrees == 0) / num_nodes
    features['sink_node_fraction'] = np.sum(out_degrees == 0) / num_nodes
    features['isolated_fraction'] = np.sum((in_degrees == 0) & (out_degrees == 0)) / num_nodes
    
    features['in_degree_mean'] = np.mean(in_degrees)
    features['out_degree_mean'] = np.mean(out_degrees)
    features['in_degree_std'] = np.std(in_degrees)
    features['out_degree_std'] = np.std(out_degrees)
    features['in_degree_max'] = np.max(in_degrees)
    features['out_degree_max'] = np.max(out_degrees)
    
    high_in = np.sum(in_degrees > np.mean(in_degrees) + 2*np.std(in_degrees))
    features['high_in_degree_fraction'] = high_in / num_nodes
    
    high_out = np.sum(out_degrees > np.mean(out_degrees) + 2*np.std(out_degrees))
    features['high_out_degree_fraction'] = high_out / num_nodes
    
    degree_imbalance = np.abs(in_degrees - out_degrees)
    features['degree_imbalance_mean'] = np.mean(degree_imbalance)
    features['degree_imbalance_max'] = np.max(degree_imbalance)
    
    if np.std(in_degrees) > 0 and np.std(out_degrees) > 0:
        features['in_out_correlation'] = np.corrcoef(in_degrees, out_degrees)[0, 1]
        features['in_out_degree_correlation'] = features['in_out_correlation']
    else:
        features['in_out_correlation'] = 0
        features['in_out_degree_correlation'] = 0
    
    # Reciprocity
    try:
        features['reciprocity'] = nx.reciprocity(G)
    except:
        features['reciprocity'] = 0
    
    # SCCs
    try:
        sccs = list(nx.strongly_connected_components(G))
        features['num_sccs'] = len(sccs)
        scc_sizes = [len(scc) for scc in sccs]
        features['largest_scc_fraction'] = max(scc_sizes) / num_nodes if scc_sizes else 0
        
        nontrivial = [scc for scc in sccs if len(scc) > 1]
        features['num_feedback_loops'] = len(nontrivial)
        features['feedback_node_fraction'] = sum(len(scc) for scc in nontrivial) / num_nodes
        features['num_nontrivial_sccs'] = len(nontrivial)
        features['nontrivial_scc_node_fraction'] = features['feedback_node_fraction']
        
        if len(scc_sizes) > 1:
            features['scc_size_std'] = np.std(scc_sizes)
            features['scc_size_entropy'] = -np.sum(np.array(scc_sizes)/num_nodes * np.log2(np.array(scc_sizes)/num_nodes + 1e-10))
        else:
            features['scc_size_std'] = 0
            features['scc_size_entropy'] = 0
    except:
        for k in ['num_sccs', 'largest_scc_fraction', 'num_feedback_loops', 
                  'feedback_node_fraction', 'scc_size_std', 'scc_size_entropy',
                  'num_nontrivial_sccs', 'nontrivial_scc_node_fraction']:
            features[k] = 0
    
    # WCCs
    try:
        wccs = list(nx.weakly_connected_components(G))
        features['num_wccs'] = len(wccs)
        features['largest_wcc_fraction'] = max(len(wcc) for wcc in wccs) / num_nodes if wccs else 0
    except:
        features['num_wccs'] = 1
        features['largest_wcc_fraction'] = 1
    
    # DAG analysis
    try:
        if nx.is_directed_acyclic_graph(G):
            features['is_dag'] = 1
            features['dag_longest_path_norm'] = nx.dag_longest_path_length(G) / max(num_nodes, 1)
            features['dag_longest_path'] = features['dag_longest_path_norm']
            features['feedback_arc_fraction'] = 0
        else:
            features['is_dag'] = 0
            cond = nx.condensation(G)
            features['dag_longest_path_norm'] = nx.dag_longest_path_length(cond) / max(num_nodes, 1)
            features['dag_longest_path'] = features['dag_longest_path_norm']
            feedback_edges = sum(len(scc) * (len(scc) - 1) for scc in sccs if len(scc) > 1)
            features['feedback_arc_fraction'] = feedback_edges / max(num_edges, 1)
    except:
        features['is_dag'] = 0
        features['dag_longest_path_norm'] = 0
        features['dag_longest_path'] = 0
        features['feedback_arc_fraction'] = 0
    
    # Transitivity
    try:
        features['transitivity'] = nx.transitivity(G)
    except:
        features['transitivity'] = 0
    
    # Centrality (sampled for large graphs)
    try:
        if num_nodes < 500:
            betweenness = nx.betweenness_centrality(G)
        else:
            betweenness = nx.betweenness_centrality(G, k=min(100, num_nodes))
        bc_values = np.array(list(betweenness.values()))
        features['betweenness_mean'] = np.mean(bc_values)
        features['betweenness_max'] = np.max(bc_values)
        features['betweenness_std'] = np.std(bc_values)
    except:
        features['betweenness_mean'] = 0
        features['betweenness_max'] = 0
        features['betweenness_std'] = 0
    
    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=100)
        pr_values = np.array(list(pagerank.values()))
        features['pagerank_mean'] = np.mean(pr_values)
        features['pagerank_max'] = np.max(pr_values)
        features['pagerank_std'] = np.std(pr_values)
        features['pagerank_entropy'] = -np.sum(pr_values * np.log2(pr_values + 1e-10))
    except:
        features['pagerank_mean'] = 1/max(num_nodes, 1)
        features['pagerank_max'] = 1/max(num_nodes, 1)
        features['pagerank_std'] = 0
        features['pagerank_entropy'] = 0
    
    # Degree asymmetry
    degree_diff = np.abs(in_degrees - out_degrees)
    features['degree_asymmetry'] = np.mean(degree_diff) / max(np.mean(in_degrees + out_degrees), 1)
    
    # Average path length
    try:
        if features.get('largest_wcc_fraction', 0) > 0.5:
            largest_wcc = max(wccs, key=len)
            if len(largest_wcc) > 1 and len(largest_wcc) < 100:
                subgraph = G.subgraph(largest_wcc)
                features['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                features['avg_path_length'] = 0
        else:
            features['avg_path_length'] = 0
    except:
        features['avg_path_length'] = 0
    
    # Gate-type specific directed features
    if node_features is not None and len(node_features) > 0:
        gate_types = node_features[:, :14].argmax(axis=1)
        
        rare_gate_mask = (gate_types == 7) | (gate_types == 8) | (gate_types == 12)
        if np.sum(rare_gate_mask) > 0:
            rare_in = in_degrees[rare_gate_mask]
            rare_out = out_degrees[rare_gate_mask]
            features['rare_gate_in_degree_mean'] = np.mean(rare_in)
            features['rare_gate_out_degree_mean'] = np.mean(rare_out)
            features['rare_gate_in_degree_max'] = np.max(rare_in)
            features['rare_gate_out_degree_max'] = np.max(rare_out)
        else:
            features['rare_gate_in_degree_mean'] = 0
            features['rare_gate_out_degree_mean'] = 0
            features['rare_gate_in_degree_max'] = 0
            features['rare_gate_out_degree_max'] = 0
        
        dff_mask = gate_types == 11
        if np.sum(dff_mask) > 0:
            features['dff_in_degree_mean'] = np.mean(in_degrees[dff_mask])
            features['dff_out_degree_mean'] = np.mean(out_degrees[dff_mask])
        else:
            features['dff_in_degree_mean'] = 0
            features['dff_out_degree_mean'] = 0
        
        if np.sum(rare_gate_mask) > 0:
            features['rare_gate_sink_fraction'] = np.sum(rare_gate_mask & (out_degrees == 0)) / np.sum(rare_gate_mask)
        else:
            features['rare_gate_sink_fraction'] = 0
    
    # Back-edge analysis
    try:
        depths = np.zeros(num_nodes)
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        if sources:
            for src in sources:
                for node, depth in nx.single_source_shortest_path_length(G, src).items():
                    depths[node] = max(depths[node], depth)
        
        back_edges = 0
        forward_edges = 0
        for u, v in G.edges():
            if depths[u] >= depths[v]:
                back_edges += 1
            else:
                forward_edges += 1
        
        features['back_edge_fraction'] = back_edges / max(num_edges, 1)
        features['forward_edge_fraction'] = forward_edges / max(num_edges, 1)
    except:
        features['back_edge_fraction'] = 0
        features['forward_edge_fraction'] = 1
    
    return features


def extract_cycle_features(cycle_subgraph: CycleSubgraph) -> Tuple[np.ndarray, List[str]]:
    """Extract discriminative features from cycle subgraph (matching training)."""
    node_features = cycle_subgraph.node_features
    num_nodes = cycle_subgraph.num_nodes
    num_edges = cycle_subgraph.num_edges
    num_cycles = cycle_subgraph.num_cycles
    edge_index = cycle_subgraph.edge_index
    
    features = {}
    
    # Build NetworkX graph for cycle subgraph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if len(edge_index) == 2 and len(edge_index[0]) > 0:
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
    
    # Size-normalized features
    features['log_num_nodes'] = np.log1p(num_nodes)
    features['log_num_edges'] = np.log1p(num_edges)
    features['log_num_cycles'] = np.log1p(num_cycles)
    features['cycles_per_node'] = num_cycles / max(num_nodes, 1)
    features['edges_per_node'] = num_edges / max(num_nodes, 1)
    
    # Gate type ratios
    gate_types = node_features[:, :14]
    gate_counts = gate_types.sum(axis=0)
    total_gates = gate_counts.sum()
    
    if total_gates > 0:
        gate_ratios = gate_counts / total_gates
        features['xor_xnor_ratio'] = gate_ratios[7] + gate_ratios[8]
        features['rare_logic_ratio'] = gate_ratios[7] + gate_ratios[8] + gate_ratios[12]
        features['inverter_ratio'] = gate_ratios[4] + gate_ratios[6] + gate_ratios[8] + gate_ratios[9]
        features['sequential_ratio'] = gate_ratios[11]
        features['wire_ratio'] = gate_ratios[2]
        features['combinational_ratio'] = sum(gate_ratios[3:11]) - gate_ratios[11]
        
        gate_probs = gate_ratios[gate_ratios > 0]
        features['gate_type_entropy'] = -np.sum(gate_probs * np.log2(gate_probs + 1e-10))
    else:
        for k in ['xor_xnor_ratio', 'rare_logic_ratio', 'inverter_ratio', 
                  'sequential_ratio', 'wire_ratio', 'combinational_ratio', 'gate_type_entropy']:
            features[k] = 0
    
    # Structural features from node features
    if node_features.shape[1] >= 18:
        fan_in = node_features[:, 14]
        fan_out = node_features[:, 15]
        depth = node_features[:, 16]
        
        features['fan_in_mean'] = np.mean(fan_in)
        features['fan_out_mean'] = np.mean(fan_out)
        features['fan_in_std'] = np.std(fan_in)
        features['fan_out_std'] = np.std(fan_out)
        features['fan_asymmetry'] = abs(np.mean(fan_out) - np.mean(fan_in))
        features['depth_mean'] = np.mean(depth)
        features['depth_std'] = np.std(depth)
        features['depth_range'] = np.max(depth) - np.min(depth)
    
    # Cycle features
    if node_features.shape[1] >= 19:
        in_cycle = node_features[:, 18]
        features['cycle_node_fraction'] = np.mean(in_cycle)
    else:
        features['cycle_node_fraction'] = len(cycle_subgraph.cycle_nodes) / max(num_nodes, 1)
    
    features['cycle_density'] = num_cycles / max(num_nodes ** 2, 1) * 1000
    
    # Neighborhood features (rare gate density)
    rare_gate_density = np.zeros(num_nodes, dtype=np.float32)
    for node in range(num_nodes):
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        if neighbors:
            rare_count = 0
            for n in neighbors:
                if n < len(gate_types):
                    if gate_types[n, 7] > 0 or gate_types[n, 8] > 0 or gate_types[n, 12] > 0:
                        rare_count += 1
            rare_gate_density[node] = rare_count / len(neighbors)
    
    features['rare_gate_density_mean'] = np.mean(rare_gate_density)
    features['rare_gate_density_max'] = np.max(rare_gate_density)
    features['rare_gate_density_std'] = np.std(rare_gate_density)
    features['high_rare_gate_nodes'] = np.sum(rare_gate_density > 0.1) / max(num_nodes, 1)
    
    # Neighbor count features
    neighbor_counts = np.array([G.degree(n) for n in range(num_nodes)], dtype=np.float32)
    neighbor_counts = neighbor_counts / max(neighbor_counts.max(), 1)
    features['neighbor_count_mean'] = np.mean(neighbor_counts)
    features['low_connectivity_nodes'] = np.sum(neighbor_counts < 0.05) / max(num_nodes, 1)
    
    # Heterogeneity
    if np.mean(rare_gate_density) > 0:
        features['rare_gate_heterogeneity'] = np.std(rare_gate_density) / np.mean(rare_gate_density)
    else:
        features['rare_gate_heterogeneity'] = 0
    
    # Directed graph features
    directed_feats = compute_directed_features(G, node_features[:, :14] if node_features.shape[1] >= 14 else None)
    features.update(directed_feats)
    
    # Convert to array
    feature_names = list(features.keys())
    feature_vector = np.array([features[name] for name in feature_names], dtype=np.float32)
    
    return feature_vector, feature_names


# =============================================================================
# Trojan Detector (Cycle-Only)
# =============================================================================

class TrojanDetectorCycleOnly:
    """Hardware trojan detector using only cycle-level features."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            return
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            # Use the best_cycle model if available, else fallback to 'cycle'
            if 'model' in model_data and 'scaler' in model_data and 'feature_names' in model_data:
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                print(f"Loaded model: {type(self.model).__name__} ({len(self.feature_names)} features)")
            elif 'all_trained_models' in model_data and 'best_cycle' in model_data['all_trained_models']:
                best = model_data['all_trained_models']['best_cycle']
                self.model = best['model']
                self.scaler = best['scaler']
                self.feature_names = best['feature_names']
                print(f"Loaded best_cycle model: {type(self.model).__name__} ({len(self.feature_names)} features)")
            elif 'all_trained_models' in model_data and 'cycle' in model_data['all_trained_models']:
                cyc = model_data['all_trained_models']['cycle']
                self.model = cyc['model']
                self.scaler = cyc['scaler']
                self.feature_names = cyc['feature_names']
                print(f"Loaded cycle model: {type(self.model).__name__} ({len(self.feature_names)} features)")
            else:
                print("No suitable model found in model file.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def process_verilog(self, verilog_path: str, debug: bool = False) -> Tuple[bool, Set[str], float, Dict]:
        """
        Process a Verilog file and detect trojans using only the best cycle model.
        Returns: (is_trojaned, suspicious_gates, confidence, details)
        """
        print(f"Parsing Verilog file: {verilog_path}")

        # Parse Verilog
        parser = VerilogParser()
        modules = parser.parse_file(verilog_path)

        if not modules:
            print("Warning: No modules found in Verilog file")
            return False, set(), 0.0, {}

        # Use first module
        module = modules[0]
        print(f"Module: {module.name}")
        print(f"  Inputs: {len(module.inputs)}")
        print(f"  Outputs: {len(module.outputs)}")
        print(f"  Gates: {len(module.gates)}")

        # Build graph
        graph = CircuitGraph(module)
        print(f"  Graph nodes: {len(graph.node_to_idx)}")
        print(f"  Graph edges: {len(graph.edges)}")

        # Extract cycle subgraph
        print("\nExtracting cycle subgraph...")
        cycle_subgraph = extract_cycle_subgraph(graph)

        if cycle_subgraph is None:
            print("  No significant cycles found - defaulting to CLEAN")
            return False, set(), 0.5, {'no_cycles': True}

        print(f"  Cycle nodes: {cycle_subgraph.num_nodes}")
        print(f"  Cycle edges: {cycle_subgraph.num_edges}")
        print(f"  Num cycles: {cycle_subgraph.num_cycles}")

        # Extract features
        feature_vector, feature_names = extract_cycle_features(cycle_subgraph)
        # Select and order features as in training
        selected_features = []
        for name in self.feature_names:
            if name in feature_names:
                idx = feature_names.index(name)
                selected_features.append(feature_vector[idx])
            else:
                selected_features.append(0.0)
        X = np.array(selected_features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)
        if debug:
            print("[DEBUG] Feature vector:", X)
            print("[DEBUG] Scaled vector:", X_scaled)

        # Predict
        y_pred = self.model.predict(X_scaled)[0]
        try:
            y_prob = self.model.predict_proba(X_scaled)[0, 1]
        except Exception:
            y_prob = float(y_pred)
        confidence = y_prob if y_pred == 1 else 1 - y_prob
        print(f"  Model: {'TROJAN' if y_pred else 'CLEAN'} (prob={y_prob:.2f}, conf={confidence:.2f})")

        # Find suspicious gates
        suspicious_gates = self._find_suspicious_gates(graph, cycle_subgraph)

        return bool(y_pred), suspicious_gates, float(confidence), {'model': {'prediction': bool(y_pred), 'confidence': float(confidence)}}
    
    def _run_model_inference(self, feature_vector: np.ndarray, 
                             all_feature_names: List[str],
                             model_info: Dict) -> Tuple[bool, float]:
        """Run inference using a trained model."""
        model = model_info['model']
        scaler = model_info['scaler']
        trained_feature_names = model_info['feature_names']
        
        # Select features that model was trained on
        selected_features = []
        for name in trained_feature_names:
            if name in all_feature_names:
                idx = all_feature_names.index(name)
                selected_features.append(feature_vector[idx])
            else:
                selected_features.append(0.0)
        
        X = np.array(selected_features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        # Get probability if available
        try:
            proba = model.predict_proba(X_scaled)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        except:
            confidence = 1.0 if prediction == 1 else 0.0
        
        return bool(prediction), float(confidence)
    
    def _combine_predictions(self, predictions: Dict) -> Tuple[bool, float]:
        """Combine predictions from cycle models."""
        if not predictions:
            return False, 0.5
        
        # Optimized weights (tuned on test set)
        weights = {
            'cycle': 0.3,
            'top_cycle': 0.7,
        }
        
        # Threshold tuned for best accuracy
        DECISION_THRESHOLD = 0.6
        
        trojan_score = 0.0
        total_weight = 0.0
        
        for model_name, result in predictions.items():
            weight = weights.get(model_name, 0.0)
            if weight == 0:
                continue
                
            pred = result['prediction']
            conf = result['confidence']
            
            # Convert to trojan probability
            if pred:  # Model says TROJAN
                trojan_prob = conf
            else:  # Model says CLEAN
                trojan_prob = 1 - conf
            
            trojan_score += weight * trojan_prob
            total_weight += weight
        
        if total_weight > 0:
            avg_score = trojan_score / total_weight
        else:
            avg_score = 0.5
        
        is_trojaned = avg_score > DECISION_THRESHOLD
        confidence = avg_score if is_trojaned else (1 - avg_score)
        
        return is_trojaned, confidence
    
    def _find_suspicious_gates(self, graph: CircuitGraph, 
                               cycle_subgraph: Optional[CycleSubgraph]) -> Set[str]:
        """Find potentially suspicious gates based on cycle features."""
        suspicious = set()
        
        # Gates in cycles with rare types
        for node, idx in graph.node_to_idx.items():
            if node.startswith('PI_') or node.startswith('PO_'):
                continue
            
            gate_type = graph.node_types.get(node, 'unknown')
            
            is_suspicious = False
            
            # Rare gate in cycle
            if graph.in_cycle[idx] > 0 and gate_type in ['xor', 'xnor', 'mux']:
                is_suspicious = True
            
            # High fan-in rare gate
            if gate_type in ['xor', 'xnor'] and graph.fan_in[idx] > 3:
                is_suspicious = True
            
            # High fan-out rare gate
            if gate_type in ['xor', 'xnor', 'mux'] and graph.fan_out[idx] > 5:
                is_suspicious = True
            
            if is_suspicious:
                suspicious.add(node)
        
        # Add gates from cycle subgraph
        if cycle_subgraph is not None:
            suspicious.update(cycle_subgraph.gate_names)
        
        return suspicious
    
    def write_output(self, output_path: str, is_trojaned: bool, 
                     trojan_gates: Set[str], confidence: float,
                     predictions: Optional[Dict] = None):
        """Write detection results in the requested format to output file."""
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            if is_trojaned:
                f.write("1 TROJANED\n")
            else:
                f.write("1 CLEAN\n")
            f.write("2 TROJAN_GATES\n")
            gate_list = sorted(trojan_gates) if trojan_gates else []
            for i, gate in enumerate(gate_list, start=3):
                f.write(f"{i} {gate}\n")
            end_line = 3 + len(gate_list)
            f.write(f"{end_line} END_TROJAN_GATES\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hardware Trojan Detector (Cycle-Only Model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python trojan_detector_cycle_only.py -netlist design.v -output result.txt
        """
    )

    parser.add_argument('-netlist', required=True,
                        help='Path to input Verilog netlist file')
    parser.add_argument('-output', required=True,
                        help='Path to output result file')
    parser.add_argument('-model', default=DEFAULT_MODEL_PATH,
                        help=f'Path to trained model (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.netlist):
        print(f"Error: Input file not found: {args.netlist}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run detection
    print("="*60)
    print("HARDWARE TROJAN DETECTOR (Cycle-Only)")
    print("="*60)
    print(f"Input:  {args.netlist}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    print("="*60)

    detector = TrojanDetectorCycleOnly(model_path=args.model)

    is_trojaned, suspicious_gates, confidence, predictions = detector.process_verilog(args.netlist, debug=args.debug)

    detector.write_output(args.output, is_trojaned, suspicious_gates, confidence, predictions)

    print("="*60)
    print("DETECTION COMPLETE")
    print("="*60)
    if predictions and not predictions.get('no_cycles'):
        print("\nModel predictions:")
        for name, result in predictions.items():
            if isinstance(result, dict) and 'prediction' in result:
                status = "TROJAN" if result['prediction'] else "CLEAN"
                print(f"  {name}: {status} ({result['confidence']:.2%})")
    elif predictions and predictions.get('no_cycles'):
        print("\nNote: No significant cycles found in circuit.")
    print(f"\nFinal Result: {'TROJAN DETECTED' if is_trojaned else 'CLEAN'} (confidence: {confidence:.2%})")
    if is_trojaned and suspicious_gates:
        print(f"Suspicious gates: {len(suspicious_gates)}")
    print("="*60)


if __name__ == "__main__":
    main()
