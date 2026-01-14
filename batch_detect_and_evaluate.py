#!/usr/bin/env python3
"""
Complete Trojan Detector Pipeline System

This script combines:
1. Batch processing of design files through a detector
2. Evaluation against ground truth with detailed metrics
3. Comprehensive scoring with gate detection analysis

Usage:
    python complete_pipeline.py                        # Run everything
    python complete_pipeline.py --detect-only         # Only run detection
    python complete_pipeline.py --evaluate-only       # Only run evaluation
    python complete_pipeline.py --start 0 --end 60    # Process specific range

Scoring System:
- 2 points for correct detection (TROJANED/CLEAN classification)
- Additional F1 score points (0-1) for trojaned cases with gate detection
- 0 points if detection is wrong (false positive or false negative)
- Total score = sum of all case scores
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import time


class CompletePipeline:
    """Complete pipeline for Trojan detection and evaluation."""
    
    def __init__(self, detector_script: str = "trojan_detector_wrapper.py"):
        self.detector_script = Path(detector_script)
        self.python_exe = sys.executable
        
        # Verify detector script exists
        if not self.detector_script.exists():
            print(f"WARNING: Detector script not found: {self.detector_script}")
            print("You'll need to specify --detector-script or implement detection logic")
    
    def parse_result_file(self, filepath: Path) -> Tuple[bool, Set[str]]:
        """
        Parse a result file to extract trojaned status and gate names.
        
        Returns:
            (is_trojaned, set_of_gate_names)
        """
        if not filepath.exists():
            return False, set()
        
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            
            # Check first line for trojaned status
            first_line = lines[0] if lines else ""
            is_trojaned = "TROJANED" in first_line
            
            # Extract gate names between TROJAN_GATES and END_TROJAN_GATES
            gates = set()
            in_gates_section = False
            
            for line in lines[1:]:
                if "TROJAN_GATES" in line and "END" not in line:
                    in_gates_section = True
                    continue
                elif "END_TROJAN_GATES" in line:
                    in_gates_section = False
                    continue
                elif in_gates_section and line:
                    # Extract gate name (skip line numbers)
                    parts = line.split()
                    if len(parts) >= 1:
                        # Last part is typically the gate name
                        gate_name = parts[-1]
                        if gate_name and not gate_name[0].isdigit():
                            gates.add(gate_name)
            
            return is_trojaned, gates
        
        except Exception as e:
            print(f"  Error parsing {filepath}: {e}")
            return False, set()
    
    def calculate_f1_score(self, predicted_gates: Set[str], 
                          actual_gates: Set[str]) -> Tuple[float, int, int, int, float, float]:
        """
        Calculate F1 score for trojan gate detection.
        
        Returns:
            (f1_score, true_positives, false_positives, false_negatives, precision, recall)
        """
        if not predicted_gates and not actual_gates:
            # Both empty - perfect match
            return 1.0, 0, 0, 0, 1.0, 1.0
        
        tp = len(predicted_gates & actual_gates)
        fp = len(predicted_gates - actual_gates)
        fn = len(actual_gates - predicted_gates)
        
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0, 0, 0, 0, 1.0, 1.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1, tp, fp, fn, precision, recall
    
    def evaluate_design(self, design_id: int, 
                       pred_file: Path,
                       truth_file: Path) -> Dict:
        """
        Evaluate a single design with detailed gate metrics.
        
        Returns:
            Dict with: correctness, points, f1_score, tp, fp, fn, precision, recall, gate percentages, details
        """
        # Parse files
        pred_trojaned, pred_gates = self.parse_result_file(pred_file)
        truth_trojaned, truth_gates = self.parse_result_file(truth_file)
        
        # Check correctness
        correctness = (pred_trojaned == truth_trojaned)
        
        if not correctness:
            # Wrong detection = 0 points
            return {
                'design_id': design_id,
                'correct': False,
                'points': 0.0,
                'f1_score': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'precision': 0.0,
                'recall': 0.0,
                'error_type': 'WRONG_DETECTION',
                'predicted': 'TROJANED' if pred_trojaned else 'CLEAN',
                'actual': 'TROJANED' if truth_trojaned else 'CLEAN',
                'num_pred_gates': len(pred_gates),
                'num_actual_gates': len(truth_gates),
                'gate_detection_rate': 0.0,
                'false_alarm_rate': 0.0
            }
        
        # Correct detection
        if not truth_trojaned:
            # Clean design - 2 points
            return {
                'design_id': design_id,
                'correct': True,
                'points': 2.0,
                'f1_score': 0.0,  # N/A for clean
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'precision': 0.0,  # N/A for clean
                'recall': 0.0,  # N/A for clean
                'error_type': None,
                'predicted': 'CLEAN',
                'actual': 'CLEAN',
                'num_pred_gates': 0,
                'num_actual_gates': 0,
                'gate_detection_rate': 0.0,
                'false_alarm_rate': 0.0
            }
        
        # Trojaned design - 2 points + F1 score points
        f1_score, tp, fp, fn, precision, recall = self.calculate_f1_score(pred_gates, truth_gates)
        points = 2.0 + f1_score  # 2 for correct detection + F1 score (0-1)
        
        # Calculate gate detection rate and false alarm rate
        gate_detection_rate = tp / len(truth_gates) * 100 if truth_gates else 0.0
        false_alarm_rate = fp / len(pred_gates) * 100 if pred_gates else 0.0
        
        return {
            'design_id': design_id,
            'correct': True,
            'points': points,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'error_type': None,
            'predicted': 'TROJANED',
            'actual': 'TROJANED',
            'num_pred_gates': len(pred_gates),
            'num_actual_gates': len(truth_gates),
            'gate_detection_rate': gate_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'predicted_gates': sorted(list(pred_gates)),
            'actual_gates': sorted(list(truth_gates))
        }
    
    def detect_design(self, design_id: int, design_dir: Path, 
                     output_dir: Path, verbose: bool = False) -> bool:
        """
        Run detector on a single design.
        
        Returns:
            True if successful, False otherwise
        """
        design_file = design_dir / f"design{design_id}.v"
        output_file = output_dir / f"pred{design_id}.txt"
        
        if not design_file.exists():
            print(f"  ✗ design{design_id}: File not found: {design_file}")
            return False
        
        # Check if detector script exists
        if not self.detector_script.exists():
            print(f"  ✗ design{design_id}: Detector script not found: {self.detector_script}")
            print(f"    Please implement detection logic or specify --detector-script")
            return False
        
        try:
            result = subprocess.run(
                [self.python_exe, str(self.detector_script),
                 "-netlist", str(design_file),
                 "-output", str(output_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"  ✗ design{design_id}: Detection failed")
                if verbose:
                    print(f"    STDERR: {result.stderr[:200]}")
                return False
            
            if output_file.exists():
                # Read result
                with open(output_file, 'r') as f:
                    first_line = f.readline().strip()
                
                status = "TROJANED" if "TROJANED" in first_line else "CLEAN"
                print(f"  ✓ design{design_id}: {status}")
                return True
            else:
                print(f"  ✗ design{design_id}: Output file not created")
                return False
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ design{design_id}: Timeout (>5 min)")
            return False
        except Exception as e:
            print(f"  ✗ design{design_id}: Error - {e}")
            return False
    
    def process_detection(self, design_dir: Path, output_dir: Path, 
                         start: int = 0, end: int = 60,
                         skip_ids: Set[int] = None,
                         verbose: bool = False) -> List[int]:
        """
        Process all designs in the range.
        
        Returns:
            List of successfully processed design IDs
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default skip_ids
        if skip_ids is None:
            skip_ids = set()
        
        successful = []
        failed = []
        
        print(f"\n{'='*70}")
        print(f"DETECTION PHASE: PROCESSING DESIGNS {start} to {end-1}")
        print(f"{'='*70}")
        print(f"Design Directory: {design_dir}")
        print(f"Output Directory: {output_dir}")
        if skip_ids:
            print(f"Skipping designs: {sorted(skip_ids)}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for design_id in range(start, end):
            if design_id in skip_ids:
                print(f"  - design{design_id}: Skipped")
                continue
            
            success = self.detect_design(design_id, design_dir, output_dir, verbose)
            if success:
                successful.append(design_id)
            else:
                failed.append(design_id)
        
        elapsed = time.time() - start_time
        
        # Summary
        print(f"\n{'='*70}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*70}")
        total_designs = (end - start) - len(skip_ids)
        print(f"Processed: {len(successful)}/{total_designs} designs")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Time: {elapsed:.1f} seconds ({elapsed/total_designs:.1f}s per design)")
        
        if failed:
            print(f"\nFailed designs: {sorted(failed)}")
        
        print(f"{'='*70}\n")
        
        return successful
    
    def process_evaluation(self, pred_dir: Path, truth_dir: Path, 
                          output_file: Path = None,
                          verbose: bool = False) -> Dict:
        """
        Run evaluation on predictions.
        
        Returns:
            Evaluation results dictionary
        """
        if output_file is None:
            output_file = Path("evaluation_results.json")
        
        print(f"\n{'='*90}")
        print(f"EVALUATION PHASE: COMPARING PREDICTIONS WITH GROUND TRUTH")
        print(f"{'='*90}")
        print(f"Predictions Dir: {pred_dir}")
        print(f"Ground Truth Dir: {truth_dir}")
        print(f"Output File: {output_file}")
        print(f"{'='*90}")
        
        # Verify directories exist
        if not pred_dir.exists():
            print(f"ERROR: Predictions directory not found: {pred_dir}")
            return {}
        
        if not truth_dir.exists():
            print(f"ERROR: Ground truth directory not found: {truth_dir}")
            return {}
        
        # Find all result files
        truth_files = sorted(truth_dir.glob("result*.txt"), 
                           key=lambda x: int(x.stem.replace("result", "")))
        
        if not truth_files:
            print(f"ERROR: No result files found in {truth_dir}")
            return {}
        
        print(f"\nFound {len(truth_files)} ground truth files\n")
        
        # Evaluate each design
        all_results = []
        total_points = 0.0
        correct_detections = 0
        trojaned_designs = 0
        clean_designs = 0
        
        # Track gate metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        trojaned_with_gates = 0
        total_gate_detection_rate = 0.0
        total_false_alarm_rate = 0.0
        
        for truth_file in truth_files:
            design_id = int(truth_file.stem.replace("result", ""))
            pred_file = pred_dir / f"pred{design_id}.txt"
            
            # Evaluate
            result = self.evaluate_design(design_id, pred_file, truth_file)
            all_results.append(result)
            
            # Track stats
            total_points += result['points']
            if result['correct']:
                correct_detections += 1
            
            if result['actual'] == 'TROJANED':
                trojaned_designs += 1
                if result['correct']:
                    trojaned_with_gates += 1
                    total_tp += result['tp']
                    total_fp += result['fp']
                    total_fn += result['fn']
                    total_gate_detection_rate += result['gate_detection_rate']
                    total_false_alarm_rate += result['false_alarm_rate']
            else:
                clean_designs += 1
            
            # Print result with detailed metrics
            status = "✓" if result['correct'] else "✗"
            if result['actual'] == 'TROJANED':
                if result['correct']:
                    print(f"{status} design{design_id:02d}: TROJANED")
                    print(f"    F1={result['f1_score']:.3f} | Precision={result['precision']:.3f} | "
                          f"Recall={result['recall']:.3f}")
                    print(f"    Gates: TP={result['tp']} FP={result['fp']} FN={result['fn']} | "
                          f"Detection Rate={result['gate_detection_rate']:.1f}% | "
                          f"False Alarm={result['false_alarm_rate']:.1f}%")
                    print(f"    Points: {result['points']:.2f} (2.0 base + {result['f1_score']:.3f} F1)")
                else:
                    print(f"{status} design{design_id:02d}: WRONG DETECTION")
                    print(f"    Predicted: CLEAN | Actual: TROJANED | Points: 0.00")
            else:
                if result['correct']:
                    print(f"{status} design{design_id:02d}: CLEAN (Points=2.00)")
                else:
                    print(f"{status} design{design_id:02d}: WRONG DETECTION")
                    print(f"    Predicted: TROJANED | Actual: CLEAN | Points: 0.00")
        
        # Print summary
        print("\n" + "=" * 90)
        print("SUMMARY STATISTICS")
        print("=" * 90)
        print(f"Total Designs: {len(all_results)}")
        print(f"  Trojaned: {trojaned_designs}")
        print(f"  Clean: {clean_designs}")
        print(f"\nDetection Accuracy: {correct_detections}/{len(all_results)} "
              f"({100*correct_detections/len(all_results):.1f}%)")
        print(f"\nFinal Score: {total_points:.2f} points")
        print(f"  (2 points per correct detection + F1 score for trojaned cases)")
        print("=" * 90)
        
        # Trojaned cases statistics
        trojaned_results = [r for r in all_results if r['actual'] == 'TROJANED']
        if trojaned_results:
            trojaned_correct = [r for r in trojaned_results if r['correct']]
            f1_scores = [r['f1_score'] for r in trojaned_correct]
            precisions = [r['precision'] for r in trojaned_correct]
            recalls = [r['recall'] for r in trojaned_correct]
            
            print(f"\nTrojaned Cases ({len(trojaned_results)} total):")
            print(f"  ✓ Correct Detection: {len(trojaned_correct)}/{len(trojaned_results)} "
                  f"({100*len(trojaned_correct)/len(trojaned_results):.1f}%)")
            
            if f1_scores:
                avg_f1 = sum(f1_scores) / len(f1_scores)
                avg_precision = sum(precisions) / len(precisions)
                avg_recall = sum(recalls) / len(recalls)
                
                print(f"\n  Gate Detection Metrics (for correctly detected trojaned cases):")
                print(f"    Average F1 Score: {avg_f1:.3f}")
                print(f"    Average Precision: {avg_precision:.3f}")
                print(f"    Average Recall: {avg_recall:.3f}")
                print(f"    Best F1: {max(f1_scores):.3f}")
                print(f"    Worst F1: {min(f1_scores):.3f}")
                
                # Overall gate metrics
                if trojaned_with_gates > 0:
                    overall_tp = total_tp
                    overall_fp = total_fp
                    overall_fn = total_fn
                    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
                    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
                    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
                    
                    avg_gate_detection = total_gate_detection_rate / trojaned_with_gates
                    avg_false_alarm = total_false_alarm_rate / trojaned_with_gates
                    
                    print(f"\n  Overall Gate Detection (across all trojaned cases):")
                    print(f"    Total TP: {overall_tp}")
                    print(f"    Total FP: {overall_fp}")
                    print(f"    Total FN: {overall_fn}")
                    print(f"    Overall Precision: {overall_precision:.3f}")
                    print(f"    Overall Recall: {overall_recall:.3f}")
                    print(f"    Overall F1: {overall_f1:.3f}")
                    print(f"    Average Gate Detection Rate: {avg_gate_detection:.1f}%")
                    print(f"    Average False Alarm Rate: {avg_false_alarm:.1f}%")
        
        # Clean cases statistics
        clean_results = [r for r in all_results if r['actual'] == 'CLEAN']
        if clean_results:
            clean_correct = [r for r in clean_results if r['correct']]
            print(f"\nClean Cases ({len(clean_results)} total):")
            print(f"  ✓ Correct Detection: {len(clean_correct)}/{len(clean_results)} "
                  f"({100*len(clean_correct)/len(clean_results):.1f}%)")
        
        # Error analysis
        errors = [r for r in all_results if not r['correct']]
        if errors:
            print(f"\nErrors ({len(errors)} total):")
            for error in errors:
                print(f"  design{error['design_id']:02d}: Predicted {error['predicted']}, "
                      f"Actual {error['actual']}")
        
        # Final score with enhanced metrics
        print(f"\n{'='*90}")
        print(f"FINAL RESULTS WITH GATE DETECTION METRICS")
        print(f"{'='*90}")
        print(f"\n[DETECTION CORRECTNESS]")
        print(f"  Detection Accuracy: {correct_detections/len(all_results)*100:.1f}%")
        if trojaned_results:
            trojaned_detection_rate = len(trojaned_correct) / len(trojaned_results) * 100
            print(f"  Trojaned Detection Rate: {trojaned_detection_rate:.1f}%")
        if clean_results:
            clean_detection_rate = len(clean_correct) / len(clean_results) * 100
            print(f"  Clean Detection Rate: {clean_detection_rate:.1f}%")
        
        if trojaned_with_gates > 0:
            print(f"\n[GATE DETECTION METRICS]")
            print(f"  Total True Positives: {total_tp}")
            print(f"  Total False Positives: {total_fp}")
            print(f"  Total False Negatives: {total_fn}")
            print(f"  Overall Precision: {overall_precision:.3f}")
            print(f"  Overall Recall: {overall_recall:.3f}")
            print(f"  Overall F1 Score: {overall_f1:.3f}")
            print(f"  Average Gate Detection Rate: {avg_gate_detection:.1f}%")
            print(f"  Average False Alarm Rate: {avg_false_alarm:.1f}%")
        
        print(f"\n[FINAL SCORING]")
        print(f"  Final Score: {total_points:.2f} points")
        max_possible_score = 2 * len(all_results) + trojaned_designs
        print(f"  Maximum Possible: {max_possible_score} points")
        if max_possible_score > 0:
            pct = 100 * total_points / max_possible_score
            print(f"  Percentage: {pct:.1f}%")
        print(f"{'='*90}\n")
        
        # Save detailed results
        json_results = {
            'summary': {
                'total_designs': len(all_results),
                'trojaned_designs': trojaned_designs,
                'clean_designs': clean_designs,
                'correct_detections': correct_detections,
                'detection_accuracy': correct_detections / len(all_results),
                'final_score': total_points,
                'max_possible_score': max_possible_score,
                'trojaned_detection_rate': len(trojaned_correct) / len(trojaned_results) * 100 if trojaned_results else 0,
                'clean_detection_rate': len(clean_correct) / len(clean_results) * 100 if clean_results else 0,
                'gate_metrics': {
                    'total_true_positives': total_tp,
                    'total_false_positives': total_fp,
                    'total_false_negatives': total_fn,
                    'overall_precision': overall_precision if 'overall_precision' in locals() else 0,
                    'overall_recall': overall_recall if 'overall_recall' in locals() else 0,
                    'overall_f1': overall_f1 if 'overall_f1' in locals() else 0,
                    'average_gate_detection_rate': avg_gate_detection if 'avg_gate_detection' in locals() else 0,
                    'average_false_alarm_rate': avg_false_alarm if 'avg_false_alarm' in locals() else 0
                }
            },
            'results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return json_results


def main():
    parser = argparse.ArgumentParser(
        description='Complete Trojan Detector Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python complete_pipeline.py                       # Run everything
    python complete_pipeline.py --detect-only         # Only run detection
    python complete_pipeline.py --evaluate-only       # Only run evaluation
    python complete_pipeline.py --start 0 --end 10    # Process specific range
    python complete_pipeline.py --skip 5,7,12         # Skip specific designs
        """
    )
    
    parser.add_argument('--detector-script', default='trojan_detector_wrapper.py',
                        help='Detector script (default: trojan_detector_wrapper.py)')
    parser.add_argument('--design-dir', default='release_hidden',
                        help='Directory with design files (default: release_hidden/)')
    parser.add_argument('--truth-dir', default='release_hidden',
                        help='Directory with ground truth (default: release_hidden/)')
    parser.add_argument('--output-dir', default='predictions',
                        help='Directory for prediction outputs (default: predictions/)')
    parser.add_argument('--eval-output', default='evaluation_results.json',
                        help='Evaluation output file (default: evaluation_results.json)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start design ID (default: 0)')
    parser.add_argument('--end', type=int, default=60,
                        help='End design ID exclusive (default: 60)')
    parser.add_argument('--skip', type=str, default='',
                        help='Comma-separated design IDs to skip')
    parser.add_argument('--detect-only', action='store_true',
                        help='Only run detection (skip evaluation)')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only run evaluation (skip detection)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.detect_only and args.evaluate_only:
        print("ERROR: Cannot specify both --detect-only and --evaluate-only")
        sys.exit(1)
    
    design_dir = Path(args.design_dir)
    truth_dir = Path(args.truth_dir)
    output_dir = Path(args.output_dir)
    
    # Verify directories exist
    if not design_dir.exists():
        print(f"ERROR: Design directory not found: {design_dir}")
        sys.exit(1)
    
    if not truth_dir.exists():
        print(f"ERROR: Truth directory not found: {truth_dir}")
        sys.exit(1)
    
    # Parse skip list
    skip_ids = set()
    if args.skip:
        try:
            skip_ids = set(int(x.strip()) for x in args.skip.split(','))
        except ValueError:
            print("ERROR: Invalid skip list format")
            sys.exit(1)
    
    # Create pipeline
    pipeline = CompletePipeline(detector_script=args.detector_script)
    
    print("\n" + "=" * 90)
    print("COMPLETE TROJAN DETECTOR PIPELINE")
    print("=" * 90)
    
    # Detection phase
    if not args.evaluate_only:
        print(f"\nStarting detection phase...")
        successful = pipeline.process_detection(
            design_dir, output_dir,
            start=args.start, end=args.end,
            skip_ids=skip_ids,
            verbose=args.verbose
        )
        
        if not successful and not args.detect_only:
            print("WARNING: No designs were successfully processed!")
            proceed = input("Continue with evaluation? (y/n): ")
            if proceed.lower() != 'y':
                print("Aborting evaluation.")
                sys.exit(0)
    
    # Evaluation phase
    if not args.detect_only:
        print(f"\nStarting evaluation phase...")
        results = pipeline.process_evaluation(
            output_dir, truth_dir,
            output_file=Path(args.eval_output),
            verbose=args.verbose
        )
    
    print("\nPipeline completed successfully!")
    print("=" * 90)


if __name__ == "__main__":
    main()