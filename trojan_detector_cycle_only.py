#!/usr/bin/env python3
"""
Modified trojan_detector_cycle_only.py that outputs detected trojan gates
in the format required for evaluation.
"""

import os
import sys
import subprocess
from pathlib import Path

# Run the original detector and capture output
DETECTOR_SCRIPT = Path(__file__).parent / "trojan_detector_cycle_only_original.py"

def main():
    if len(sys.argv) < 5:
        print("Usage: python trojan_detector_cycle_only.py -netlist <file> -output <file>")
        sys.exit(1)
    
    args_dict = {}
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i].startswith('-'):
            args_dict[sys.argv[i]] = sys.argv[i+1] if i+1 < len(sys.argv) else None
    
    netlist_path = args_dict.get('-netlist')
    output_path = args_dict.get('-output')
    
    if not netlist_path or not output_path:
        print("Error: -netlist and -output arguments required")
        sys.exit(1)
    
    if not os.path.exists(netlist_path):
        print(f"Error: Netlist file not found: {netlist_path}")
        sys.exit(1)
    
    # Import detection logic
    sys.path.insert(0, str(Path(__file__).parent))
    
    from trojan_detector_cycle_only_original import (
        VerilogParser, CircuitGraph, extract_cycle_subgraph, 
        extract_cycle_features, TrojanDetectorCycleOnly
    )
    
    # Run detection
    detector = TrojanDetectorCycleOnly()
    is_trojaned, suspicious_gates, confidence, predictions = detector.process_verilog(netlist_path, debug=False)
    
    # Write output in required format
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if is_trojaned:
            f.write("1 TROJANED\n")
            f.write("2 TROJAN_GATES\n")
            
            # Sort gates for consistent output
            gate_list = sorted(suspicious_gates) if suspicious_gates else []
            for i, gate in enumerate(gate_list, start=3):
                f.write(f"{i} {gate}\n")
            
            end_line = 3 + len(gate_list)
            f.write(f"{end_line} END_TROJAN_GATES\n")
        else:
            f.write("1 NO_TROJAN\n")
    
    print(f"Result written to: {output_path}")

if __name__ == "__main__":
    main()