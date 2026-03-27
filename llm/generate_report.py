"""
Generates a clinical text report from pancreas segmentation results.

Supports two backends:
  - ollama: local open-weights model (default, e.g. biomistral)
  - claude: Anthropic Claude API (requires ANTHROPIC_API_KEY)

Usage:
    python generate_report.py                        # uses ollama + biomistral
    python generate_report.py --backend claude       # uses Claude API
"""

import json
import argparse


OLLAMA_MODEL = "biomistral"
CLAUDE_MODEL = "claude-opus-4-6"


def build_prompt(metrics: dict) -> str:
    return f"""You are an expert radiologist. Based on the following automatic pancreas segmentation metrics from an abdominal CT scan, generate a brief structured radiology report.

Segmentation data:
- Estimated pancreas volume: {metrics.get('vol_pred_ml', 'N/A')} ml
- Dice score: {metrics.get('dice', 'N/A')}
- Case ID: {metrics.get('case', 'N/A')}

The report should include:
1. Dimensional description of the pancreas
2. Note on segmentation quality
3. Brief conclusion

Use appropriate medical language."""


def generate_report_ollama(metrics: dict, model: str = OLLAMA_MODEL) -> str:
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(metrics)}]
    )
    return response["message"]["content"]


def generate_report_claude(metrics: dict, model: str = CLAUDE_MODEL) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": build_prompt(metrics)}]
    )
    return message.content[0].text


def generate_report(metrics: dict, backend: str = "ollama") -> str:
    if backend == "ollama":
        return generate_report_ollama(metrics)
    elif backend == "claude":
        return generate_report_claude(metrics)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'ollama' or 'claude'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama", "claude"],
                        help="LLM backend to use (default: ollama)")
    args = parser.parse_args()

    example_metrics = {
        "case":         "pancreas_001",
        "dice":         0.82,
        "vol_pred_ml":  78.5,
        "vol_gt_ml":    75.2,
        "vol_error_ml": 3.3,
    }

    print(f"=== Generating report (backend: {args.backend}) ===")
    print(f"Metrics: {json.dumps(example_metrics, indent=2)}\n")

    report = generate_report(example_metrics, backend=args.backend)
    print("=== Generated report ===")
    print(report)
