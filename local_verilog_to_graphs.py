"""
Local Script: Convert Verilog Files to Graph Data (Self-Contained)

Run this locally to convert all Verilog designs to graph format.
This version includes all necessary functions and optimized cycle detection.

Usage:
    python local_verilog_to_graphs.py
    python local_verilog_to_graphs.py --single 9    # Process only design 9
    python local_verilog_to_graphs.py --skip 9,15   # Skip designs 9 and 15

Output:
    - all_graphs.json (graph data)
    - all_graphs_metadata.json (design info, trojan gates, labels)
"""

import os
import sys
import json
import re
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not installed. Cycle detection will be limited.")

SCRIPT_DIR = Path(__file__).parent

# =============================================================================
# Gate Type Constants
# =============================================================================

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
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerilogModule:
    """Represents a parsed Verilog module."""
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    wires: List[str] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)


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
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
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
                inputs = self._parse_declaration(line, 'input')
                module.inputs.extend(inputs)
            elif line.startswith('output'):
                outputs = self._parse_declaration(line, 'output')
                module.outputs.extend(outputs)
            elif line.startswith('wire'):
                wires = self._parse_declaration(line, 'wire')
                module.wires.extend(wires)
            else:
                gate = self._parse_gate(line)
                if gate:
                    module.gates.append(gate)
                    
        return module
    
    def _parse_declaration(self, line: str, decl_type: str) -> List[str]:
        """Parse input/output/wire declarations."""
        signals = []
        line = re.sub(rf'^{decl_type}\s+', '', line)
        
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
        
        dff_pattern = r'^dff\s+(\w+)\s*\((.+)\)'
        dff_match = re.match(dff_pattern, line, re.DOTALL)
        if dff_match:
            gate_name = dff_match.group(1)
            connections = dff_match.group(2)
            gate = Gate(name=gate_name, gate_type='dff')
            
            port_pattern = r'\.(\w+)\s*\(([^)]*)\)'
            ports = re.findall(port_pattern, connections)
            
            for port_name, signal in ports:
                signal = signal.strip()
                if port_name in ['D', 'CK', 'RN', 'SN']:
                    if signal and signal != "1'b1":
                        gate.inputs.append(signal)
                    gate.parameters[port_name] = signal
                elif port_name == 'Q':
                    if signal:
                        gate.outputs.append(signal)
                    gate.parameters[port_name] = signal
            return gate
        
        standard_gate_pattern = r'^(\w+)\s+(\w+)\s*\(([^)]+)\)'
        std_match = re.match(standard_gate_pattern, line)
        if std_match:
            gate_type = std_match.group(1).lower()
            gate_name = std_match.group(2)
            connections = std_match.group(3)
            
            if gate_type in ['input', 'output', 'wire', 'reg', 'module']:
                return None
                
            gate = Gate(name=gate_name, gate_type=gate_type)
            signals = [s.strip() for s in connections.split(',') if s.strip()]
            
            if signals:
                gate.outputs.append(signals[0])
                gate.inputs.extend(signals[1:])
                
            return gate
            
        return None


# =============================================================================
# Circuit Graph
# =============================================================================

class CircuitGraph:
    """Represents a circuit as a graph for GNN processing."""
    
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
        """Build the graph representation from the module."""
        node_idx = 0
        
        # Primary inputs
        for inp in self.module.inputs:
            node_name = f"PI_{inp}"
            self.node_to_idx[node_name] = node_idx
            self.idx_to_node[node_idx] = node_name
            self.node_types[node_name] = 'input'
            self.signal_driver[inp] = node_name
            node_idx += 1
            
        # Primary outputs
        for out in self.module.outputs:
            node_name = f"PO_{out}"
            self.node_to_idx[node_name] = node_idx
            self.idx_to_node[node_idx] = node_name
            self.node_types[node_name] = 'output'
            self.signal_consumers[out].append(node_name)
            node_idx += 1
                
        # Gates
        for gate in self.module.gates:
            gate_node = gate.name
            self.node_to_idx[gate_node] = node_idx
            self.idx_to_node[node_idx] = gate_node
            self.node_types[gate_node] = gate.gate_type
            node_idx += 1
            
            for out_signal in gate.outputs:
                self.signal_driver[out_signal] = gate_node
            for in_signal in gate.inputs:
                self.signal_consumers[in_signal].append(gate_node)
        
        # Create edges
        all_signals = set(self.signal_driver.keys()) | set(self.signal_consumers.keys())
        for signal in all_signals:
            driver = self.signal_driver.get(signal)
            consumers = self.signal_consumers.get(signal, [])
            
            if driver and driver in self.node_to_idx:
                driver_idx = self.node_to_idx[driver]
                for consumer in consumers:
                    if consumer in self.node_to_idx:
                        consumer_idx = self.node_to_idx[consumer]
                        self.edges.append((driver_idx, consumer_idx))
        
        self._compute_structural_features()
    
    def _compute_structural_features(self):
        """Compute structural features with OPTIMIZED cycle detection."""
        num_nodes = len(self.node_to_idx)
        
        self.in_cycle = np.zeros(num_nodes, dtype=np.float32)
        self.fan_in = np.zeros(num_nodes, dtype=np.float32)
        self.fan_out = np.zeros(num_nodes, dtype=np.float32)
        self.depth = np.zeros(num_nodes, dtype=np.float32)
        
        # Compute fan-in and fan-out
        for src, dst in self.edges:
            self.fan_out[src] += 1
            self.fan_in[dst] += 1
        
        max_fan_in = max(self.fan_in.max(), 1)
        max_fan_out = max(self.fan_out.max(), 1)
        self.fan_in_norm = self.fan_in / max_fan_in
        self.fan_out_norm = self.fan_out / max_fan_out
        
        self.graph_features = {}
        
        if NETWORKX_AVAILABLE and num_nodes > 0:
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(self.edges)
            
            # OPTIMIZED: Use SCCs instead of simple_cycles (MUCH faster!)
            # Nodes in non-trivial SCCs are in cycles
            try:
                sccs = list(nx.strongly_connected_components(G))
                non_trivial_sccs = [scc for scc in sccs if len(scc) > 1]
                nodes_in_cycles = set()
                for scc in non_trivial_sccs:
                    nodes_in_cycles.update(scc)
                
                for node_idx in nodes_in_cycles:
                    self.in_cycle[node_idx] = 1.0
                
                self.graph_features['num_cycles'] = len(non_trivial_sccs)  # Approximate
                self.graph_features['num_nodes_in_cycles'] = len(nodes_in_cycles)
                self.graph_features['num_sccs'] = len(sccs)
                self.graph_features['num_non_trivial_sccs'] = len(non_trivial_sccs)
                self.graph_features['largest_scc_size'] = max(len(scc) for scc in sccs) if sccs else 0
            except Exception as e:
                print(f"  Warning: SCC computation failed: {e}")
                self.graph_features['num_cycles'] = 0
                self.graph_features['num_nodes_in_cycles'] = 0
                self.graph_features['num_sccs'] = 0
                self.graph_features['num_non_trivial_sccs'] = 0
                self.graph_features['largest_scc_size'] = 0
            
            # Density
            self.graph_features['density'] = nx.density(G)
            
            # Average path length (skip for large graphs)
            self.graph_features['avg_path_length'] = 0
            
            # Compute depth from primary inputs
            pi_nodes = [idx for node, idx in self.node_to_idx.items() if node.startswith('PI_')]
            if pi_nodes:
                for pi in pi_nodes:
                    try:
                        lengths = nx.single_source_shortest_path_length(G, pi)
                        for node, length in lengths.items():
                            self.depth[node] = max(self.depth[node], length)
                    except:
                        pass
            
            max_depth = max(self.depth.max(), 1)
            self.depth_norm = self.depth / max_depth
            
        else:
            self.graph_features['num_cycles'] = 0
            self.graph_features['num_nodes_in_cycles'] = 0
            self.graph_features['num_sccs'] = 0
            self.graph_features['num_non_trivial_sccs'] = 0
            self.graph_features['largest_scc_size'] = 0
            self.graph_features['density'] = len(self.edges) / max(num_nodes * (num_nodes - 1), 1)
            self.graph_features['avg_path_length'] = 0
            self.depth_norm = np.zeros(num_nodes, dtype=np.float32)
        
        self.graph_features['num_nodes'] = num_nodes
        self.graph_features['num_edges'] = len(self.edges)
        self.graph_features['avg_fan_in'] = float(np.mean(self.fan_in)) if num_nodes > 0 else 0
        self.graph_features['avg_fan_out'] = float(np.mean(self.fan_out)) if num_nodes > 0 else 0
        self.graph_features['max_fan_in'] = float(np.max(self.fan_in)) if num_nodes > 0 else 0
        self.graph_features['max_fan_out'] = float(np.max(self.fan_out)) if num_nodes > 0 else 0
                    
    def get_node_features(self) -> np.ndarray:
        """Get one-hot encoded node features."""
        num_nodes = len(self.node_to_idx)
        features = np.zeros((num_nodes, NUM_GATE_TYPES), dtype=np.float32)
        
        for node, idx in self.node_to_idx.items():
            node_type = self.node_types.get(node, 'unknown')
            type_idx = GATE_TYPES.get(node_type, GATE_TYPES['unknown'])
            features[idx, type_idx] = 1.0
            
        return features
    
    def get_enhanced_node_features(self) -> np.ndarray:
        """Get enhanced node features (18 dims)."""
        base_features = self.get_node_features()
        enhanced = np.column_stack([
            base_features,
            self.in_cycle.reshape(-1, 1),
            self.fan_in_norm.reshape(-1, 1),
            self.fan_out_norm.reshape(-1, 1),
            self.depth_norm.reshape(-1, 1)
        ])
        return enhanced.astype(np.float32)
    
    def get_edge_features(self) -> np.ndarray:
        """Get edge features (3 dims)."""
        if not self.edges:
            return np.zeros((0, 3), dtype=np.float32)
        
        edge_features = []
        for src, dst in self.edges:
            fan_out_src = self.fan_out_norm[src]
            fan_in_dst = self.fan_in_norm[dst]
            is_feedback = 1.0 if self.depth[src] > self.depth[dst] else 0.0
            edge_features.append([fan_out_src, fan_in_dst, is_feedback])
        
        return np.array(edge_features, dtype=np.float32)
    
    def get_graph_features(self) -> np.ndarray:
        """Get graph-level features (10 dims)."""
        num_nodes = max(self.graph_features['num_nodes'], 1)
        
        features = np.array([
            self.graph_features['num_cycles'] / num_nodes,
            self.graph_features['num_nodes_in_cycles'] / num_nodes,
            self.graph_features['num_non_trivial_sccs'] / max(num_nodes / 10, 1),
            self.graph_features['largest_scc_size'] / num_nodes,
            self.graph_features['density'],
            self.graph_features['avg_path_length'] / max(num_nodes, 1),
            self.graph_features['avg_fan_in'] / 10,
            self.graph_features['avg_fan_out'] / 10,
            self.graph_features['max_fan_in'] / 20,
            self.graph_features['max_fan_out'] / 20,
        ], dtype=np.float32)
        
        return features
    
    def get_edge_index(self) -> np.ndarray:
        """Get edge index in COO format."""
        if not self.edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(self.edges, dtype=np.int64).T


# =============================================================================
# Processing Functions
# =============================================================================

def process_verilog_file(filepath: str, label: Optional[int] = None) -> List[CircuitGraph]:
    """Process a single Verilog file and return circuit graphs."""
    parser = VerilogParser()
    modules = parser.parse_file(filepath)
    
    graphs = []
    for module in modules:
        graph = CircuitGraph(module)
        graph.label = label
        graphs.append(graph)
        
    return graphs


def parse_result_file(filepath: str) -> Tuple[bool, Set[str]]:
    """Parse a resultX.txt file to get trojan status and gates."""
    trojan_gates = set()
    is_trojaned = False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        lines = content.split('\n')
        
        if lines and lines[0].strip() == 'TROJANED':
            is_trojaned = True
            
            in_gates_section = False
            for line in lines[1:]:
                line = line.strip()
                if line == 'TROJAN_GATES':
                    in_gates_section = True
                elif line == 'END_TROJAN_GATES':
                    in_gates_section = False
                elif in_gates_section and line:
                    trojan_gates.add(line)
                    
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return is_trojaned, trojan_gates


def graph_to_serializable(graph: CircuitGraph, design_id: int, 
                          is_trojaned: bool, trojan_gates: Set[str]) -> dict:
    """Convert a CircuitGraph to a serializable dictionary."""
    return {
        'design_id': design_id,
        'module_name': graph.module.name,
        'is_trojaned': is_trojaned,
        'trojan_gates': list(trojan_gates),
        'num_nodes': len(graph.node_to_idx),
        'num_edges': len(graph.edges),
        'node_features': graph.get_enhanced_node_features().tolist(),
        'edge_index': graph.get_edge_index().tolist(),
        'edge_features': graph.get_edge_features().tolist(),
        'graph_features': graph.get_graph_features().tolist(),
        'node_to_idx': graph.node_to_idx,
        'idx_to_node': {str(k): v for k, v in graph.idx_to_node.items()},
        'node_types': graph.node_types,
        'edges': graph.edges,
        'depth': graph.depth.tolist(),
        'fan_in': graph.fan_in.tolist(),
        'fan_out': graph.fan_out.tolist(),
        'fan_in_norm': graph.fan_in_norm.tolist(),
        'fan_out_norm': graph.fan_out_norm.tolist(),
        'depth_norm': graph.depth_norm.tolist(),
        'in_cycle': graph.in_cycle.tolist(),
    }


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to convert all Verilog designs to graph data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Verilog to graph data')
    parser.add_argument('--start', type=int, default=0, help='Start design index')
    parser.add_argument('--end', type=int, default=30, help='End design index (exclusive)')
    parser.add_argument('--skip', type=str, default='', help='Comma-separated designs to skip')
    parser.add_argument('--single', type=int, default=None, help='Process only one design')
    args = parser.parse_args()
    
    # Configuration
    BASE_DIR = SCRIPT_DIR / "release_all(20250728)"
    OUTPUT_DIR = SCRIPT_DIR
    trojan_dir = BASE_DIR / "trojan"
    trojan_free_dir = BASE_DIR / "trojan_free"
    
    # Parse skip list
    skip_designs = set()
    if args.skip:
        skip_designs = set(int(x.strip()) for x in args.skip.split(',') if x.strip())
    
    # Determine design range
    if args.single is not None:
        design_range = [args.single]
    else:
        design_range = [i for i in range(args.start, args.end) if i not in skip_designs]
    
    print("="*70)
    print("LOCAL VERILOG TO GRAPH CONVERTER (Self-Contained)")
    print("="*70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing designs: {design_range}")
    if skip_designs:
        print(f"Skipping designs: {sorted(skip_designs)}")
    
    # Output file
    json_output = OUTPUT_DIR / "all_graphs.json"
    json_file = open(json_output, 'w')
    json_file.write('[\n')
    first_graph = True
    
    graph_count = 0
    metadata = {
        'designs': [],
        'total_graphs': 0,
        'trojaned_designs': 0,
        'clean_designs': 0,
    }
    
    for i in design_range:
        # Determine paths
        if i < 20:
            verilog_path = trojan_dir / f'design{i}.v'
            result_path = trojan_dir / f'result{i}.txt'
            source = "TROJAN"
        else:
            verilog_path = trojan_free_dir / f'design{i}.v'
            result_path = None
            source = "TROJAN-FREE"
        
        if not verilog_path.exists():
            print(f"\n  design{i}.v not found in {source} dir, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {source} design {i}")
        print(f"{'='*70}")
        
        # Get label
        if result_path and result_path.exists():
            is_trojaned, trojan_gates = parse_result_file(str(result_path))
        else:
            is_trojaned, trojan_gates = False, set()
        
        print(f"  trojaned={is_trojaned}, trojan_gates={len(trojan_gates)}")
        
        written = 0
        try:
            graphs = process_verilog_file(str(verilog_path), label=None)
            
            for graph in graphs:
                graph_data = graph_to_serializable(graph, i, is_trojaned, trojan_gates)
                
                if not first_graph:
                    json_file.write(',\n')
                json.dump(graph_data, json_file)
                json_file.flush()
                
                print(f"  Module: {graph.module.name}, Nodes: {len(graph.node_to_idx)}, Edges: {len(graph.edges)}")
                
                first_graph = False
                graph_count += 1
                written += 1
                
                del graph_data
            
            del graphs
            gc.collect()
            
        except Exception as e:
            print(f"  Error processing design{i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Update metadata
        metadata['designs'].append({
            'design_id': i,
            'source': source.lower().replace('-', '_'),
            'is_trojaned': is_trojaned,
            'num_trojan_gates': len(trojan_gates),
            'trojan_gates': list(trojan_gates),
            'num_modules': written
        })
        
        if is_trojaned:
            metadata['trojaned_designs'] += 1
        else:
            metadata['clean_designs'] += 1
        
        print(f"  [Progress: {graph_count} graphs written]")
    
    metadata['total_graphs'] = graph_count
    
    # Close JSON array
    json_file.write('\n]')
    json_file.close()
    
    # Save metadata
    print("\n" + "="*70)
    print("SAVING OUTPUT")
    print("="*70)
    
    print(f"\nSaved JSON to: {json_output}")
    print(f"  Saved {graph_count} graphs")
    
    metadata_output = OUTPUT_DIR / "all_graphs_metadata.json"
    print(f"\nSaving metadata to: {metadata_output}")
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal graphs extracted: {metadata['total_graphs']}")
    print(f"Trojaned designs: {metadata['trojaned_designs']}")
    print(f"Clean designs: {metadata['clean_designs']}")
    
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print(f"\n1. {json_output}")
    print(f"2. {metadata_output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
