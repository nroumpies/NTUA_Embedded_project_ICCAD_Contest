#!/usr/bin/env python3
"""
Wrapper script to integrate with existing trojan_detector_cycle_only.py
This script calls the detector and reformats output to include gate names.


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
from pathlib import Path

# =============================================================================
# STEP 1: Import the original detector components
# =============================================================================

# You need to keep these imports and classes from the original file
# For now, we'll create a minimal version that extracts suspicious gates

@dataclass
class Gate:
    name: str
    gate_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

@dataclass
class VerilogModule:
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    wires: List[str] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)

@dataclass
class CycleSubgraph:
    num_nodes: int
    num_edges: int
    num_cycles: int
    node_features: np.ndarray
    edge_index: List[List[int]]
    cycle_nodes: Set[int]
    gate_names: Set[str]


# Import VerilogParser and CircuitGraph from your original detector
# Assuming they're in trojan_detector_cycle_only_original.py
try:
    from trojan_detector_cycle_only_original import (
        VerilogParser, CircuitGraph, extract_cycle_subgraph,
        extract_cycle_features, TrojanDetectorCycleOnly,
        GATE_TYPES, NUM_GATE_TYPES
    )
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from original detector: {e}")
    print("Make sure you have renamed trojan_detector_cycle_only.py to trojan_detector_cycle_only_original.py")
    ORIGINAL_AVAILABLE = False


# =============================================================================
# STEP 2: Enhanced gate detection
# =============================================================================

class GateDetector:
    """Enhanced gate detection for suspicious circuits."""
    
    @staticmethod
    def find_suspicious_gates(graph: 'CircuitGraph', 
                            cycle_subgraph: Optional['CycleSubgraph'],
                            is_trojaned: bool) -> Set[str]:
        """
        Find gates suspected to be trojan gates.
        
        Strategy:
        1. All gates in cycle subgraph (feedback loops where trojans hide)
        2. Rare gate types (XOR, XNOR, MUX) in cycles with high connectivity
        3. Gates with unusual fan-in/fan-out patterns
        """
        suspicious = set()
        
        if cycle_subgraph is None:
            return suspicious
        
        # All gates in the cycle subgraph are potentially suspicious
        suspicious.update(cycle_subgraph.gate_names)
        
        # Additional heuristics for detecting trojans
        for node_name, node_idx in graph.node_to_idx.items():
            if node_name.startswith('PI_') or node_name.startswith('PO_'):
                continue
            
            gate_type = graph.node_types.get(node_name, 'unknown').lower()
            
            # Rule 1: Rare gates in cycles with high fan-in/out
            if gate_type in ['xor', 'xnor', 'mux']:
                if graph.in_cycle[node_idx] > 0:
                    suspicious.add(node_name)
                elif graph.fan_in[node_idx] > 3 or graph.fan_out[node_idx] > 5:
                    suspicious.add(node_name)
            
            # Rule 2: Sequential elements (DFF) with unusual connectivity
            if gate_type == 'dff':
                if graph.in_cycle[node_idx] > 0 and graph.fan_in[node_idx] > 2:
                    suspicious.add(node_name)
        
        return suspicious


# =============================================================================
# STEP 3: Main detector wrapper
# =============================================================================

class TrojanDetectorWrapper:
    """Wrapper that uses the original detector but outputs gate names."""
    
    def __init__(self, model_path: str = None):
        if ORIGINAL_AVAILABLE:
            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "output",
                    "cycle_AdaBoost.pkl"
                )
            self.detector = TrojanDetectorCycleOnly(model_path)
        else:
            self.detector = None
    
    def process_verilog(self, verilog_path: str) -> Tuple[bool, Set[str], float]:
        """
        Process Verilog and detect trojans.
        
        Returns:
            (is_trojaned, suspicious_gate_names, confidence)
        """
        if not ORIGINAL_AVAILABLE or self.detector is None:
            print("ERROR: Original detector not available")
            return False, set(), 0.0
        
        # Use the original detector
        is_trojaned, suspicious_gates, confidence, _ = self.detector.process_verilog(
            verilog_path, debug=False
        )
        
        return is_trojaned, suspicious_gates, confidence
    
    def write_output(self, output_path: str, is_trojaned: bool, 
                     trojan_gates: Set[str]):
        """Write output in the required format."""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if is_trojaned:
                f.write("1 TROJANED\n")
                f.write("2 TROJAN_GATES\n")
                
                # Sort gates for consistent output
                gate_list = sorted(trojan_gates) if trojan_gates else []
                for i, gate in enumerate(gate_list, start=3):
                    f.write(f"{i} {gate}\n")
                
                end_line = 3 + len(gate_list)
                f.write(f"{end_line} END_TROJAN_GATES\n")
            else:
                f.write("1 NO_TROJAN\n")


# =============================================================================
# STEP 4: Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hardware Trojan Detector (Wrapper with Gate Detection)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python trojan_detector_wrapper.py -netlist design.v -output result.txt
        """
    )

    parser.add_argument('-netlist', required=True,
                        help='Path to input Verilog netlist file')
    parser.add_argument('-output', required=True,
                        help='Path to output result file')
    parser.add_argument('-model', default=None,
                        help='Path to trained model')
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
    print("HARDWARE TROJAN DETECTOR (Wrapper)")
    print("="*60)
    print(f"Input:  {args.netlist}")
    print(f"Output: {args.output}")
    print("="*60)

    detector = TrojanDetectorWrapper(model_path=args.model)

    is_trojaned, suspicious_gates, confidence = detector.process_verilog(args.netlist)

    detector.write_output(args.output, is_trojaned, suspicious_gates)

    print("="*60)
    print("DETECTION COMPLETE")
    print("="*60)
    print(f"Result: {'TROJANED' if is_trojaned else 'CLEAN'} (confidence: {confidence:.2%})")
    if is_trojaned and suspicious_gates:
        print(f"Suspicious gates detected: {len(suspicious_gates)}")
        if args.debug:
            print(f"  {', '.join(sorted(suspicious_gates)[:10])}")
            if len(suspicious_gates) > 10:
                print(f"  ... and {len(suspicious_gates)-10} more")
    print("="*60)


if __name__ == "__main__":
    main()