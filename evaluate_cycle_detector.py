#!/usr/bin/env python3
"""
Evaluation Script for Cycle-Only Trojan Detector

Runs the trojan_detector_cycle_only.py on all designs in release_hidden/
and compares predictions against ground truth results.

Usage:
    python evaluate_cycle_detector.py
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Configuration - paths relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
HIDDEN_DIR = SCRIPT_DIR / "release_hidden"
DETECTOR_SCRIPT = SCRIPT_DIR / "trojan_detector_cycle_only.py"
PYTHON_PATH = sys.executable  # Use the current Python interpreter


def get_ground_truth(result_file: Path) -> bool:
    """Read ground truth from result file."""
    with open(result_file, 'r') as f:
        content = f.read().strip()
    return content.startswith("TROJANED")


def get_prediction(output_file: Path) -> bool:
    """Read prediction from detector output file."""
    if not output_file.exists():
        print(f"    WARNING: Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r') as f:
        content = f.read().strip()
    return content.startswith("TROJANED")


def run_detector(design_file: Path, output_file: Path, verbose: bool = False) -> bool:
    """Run the cycle detector on a design file."""
    # Use a temporary file to get the full output from detector
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        result = subprocess.run(
            [PYTHON_PATH, str(DETECTOR_SCRIPT), 
             "-netlist", str(design_file),
             "-output", str(tmp_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"    STDERR: {result.stderr}")
                print(f"    STDOUT: {result.stdout}")
            return False
        
        # Read the first line from the detector output
        if tmp_path.exists():
            with open(tmp_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Write only TROJANED or CLEAN to the prediction file
            with open(output_file, 'w') as f:
                if "TROJANED" in first_line:
                    f.write("TROJANED\n")
                elif "CLEAN" in first_line:
                    f.write("CLEAN\n")
                else:
                    return False
            
            return True
        else:
            return False
    
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    print("=" * 70)
    print("CYCLE-ONLY TROJAN DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Detector: {DETECTOR_SCRIPT}")
    print(f"Test Dir: {HIDDEN_DIR}")
    
    # Create predictions directory if it doesn't exist
    predictions_dir = SCRIPT_DIR / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    print(f"Output Dir: {predictions_dir}")
    print("=" * 70)
    
    # Verify detector script exists
    if not DETECTOR_SCRIPT.exists():
        print(f"ERROR: Detector script not found: {DETECTOR_SCRIPT}")
        sys.exit(1)
    
    # Verify hidden directory exists
    if not HIDDEN_DIR.exists():
        print(f"ERROR: Test directory not found: {HIDDEN_DIR}")
        sys.exit(1)
    
    # Find all design files
    design_files = sorted(HIDDEN_DIR.glob("design*.v"), 
                          key=lambda x: int(x.stem.replace("design", "")))
    
    if not design_files:
        print("ERROR: No design files found!")
        sys.exit(1)
    
    print(f"\nFound {len(design_files)} design files\n")
    
    # Results tracking
    results = []
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for design_file in design_files:
        design_num = int(design_file.stem.replace("design", ""))
        result_file = HIDDEN_DIR / f"result{design_num}.txt"
        output_file = predictions_dir / f"pred{design_num}.txt"

        print(f"Processing design{design_num:02d}...", end=" ")

        # Get ground truth
        if not result_file.exists():
            print(f"  WARNING: No result file for design{design_num}")
            continue

        actual = get_ground_truth(result_file)

        # Run detector to create prediction file (only TROJANED or CLEAN)
        success = run_detector(design_file, output_file, verbose=False)
        if not success:
            print(f"  ERROR: Detector failed on design{design_num}")
            continue

        # Ensure prediction file exists
        if not output_file.exists():
            print(f"  ERROR: Output file not created: {output_file}")
            continue

        # Read prediction
        predicted = get_prediction(output_file)
        
        # Print result
        print("TROJANED" if predicted else "CLEAN")

        # Record result for metrics
        correct = predicted == actual
        results.append({
            'design': design_num,
            'actual': actual,
            'predicted': predicted,
            'correct': correct
        })

        # Update confusion matrix
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1
    
    # Verify prediction files were created
    created_files = list(predictions_dir.glob("pred*.txt"))
    print(f"\n{len(created_files)} prediction files created in {predictions_dir}")
    
    # Print summary
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                TROJAN    CLEAN")
    print(f"  Actual TROJAN   {tp:3d}      {fn:3d}")
    print(f"  Actual CLEAN    {fp:3d}      {tn:3d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:    {correct}/{total} = {correct/total:.1%}")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision:   {precision:.1%}")
    else:
        print(f"  Precision:   N/A (no positive predictions)")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall:      {recall:.1%}")
    else:
        print(f"  Recall:      N/A (no actual positives)")
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)
        print(f"  Specificity: {specificity:.1%}")
    
    if tp + fp > 0 and tp + fn > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"  F1 Score:    {f1:.3f}")
    
    # List errors
    fn_list = [r['design'] for r in results if r['actual'] and not r['predicted']]
    fp_list = [r['design'] for r in results if not r['actual'] and r['predicted']]
    
    if fn_list:
        print(f"\nFalse Negatives (missed trojans): {fn_list}")
    if fp_list:
        print(f"False Positives (false alarms):   {fp_list}")
    
    print("\n" + "=" * 70)
    print(f"Prediction files saved to: {predictions_dir}")
    print("=" * 70)
    
    return correct == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)