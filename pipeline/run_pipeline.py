"""
End-to-end pipeline: pancreas segmentation -> LLM report generation.

Usage:
    python run_pipeline.py --input <ct_folder> --seg <segmentation_folder> --output <output_folder>
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from segmentation.evaluate import evaluate_folder
from llm.generate_report import generate_report


def run_pipeline(input_dir: Path, seg_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: evaluate segmentations produced by nnUNet
    print("=== Step 1: Evaluating segmentations ===")
    gt_dir  = input_dir / "labels"
    results = evaluate_folder(seg_dir, gt_dir, label=1)

    # Step 2: generate a report for each case
    print("\n=== Step 2: Generating LLM reports ===")
    for metrics in results:
        report      = generate_report(metrics)
        report_path = output_dir / f"{metrics['case']}_report.txt"
        report_path.write_text(report)
        print(f"  Saved: {report_path.name}")

    print(f"\nDone. {len(results)} reports saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Folder with CT scans and labels")
    parser.add_argument("--seg",    required=True, help="Folder with nnUNet segmentation outputs")
    parser.add_argument("--output", required=True, help="Output folder for reports")
    args = parser.parse_args()

    run_pipeline(Path(args.input), Path(args.seg), Path(args.output))
