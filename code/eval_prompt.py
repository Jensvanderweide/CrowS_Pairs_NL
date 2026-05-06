import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from code.utils import compare_pair_prompt_batch

TEMPLATES = {
    1: "waarschijnlijker",
    2: "natuurlijker",
    3: "aannemelijker",
}

def _run_single_template(lm, eval_data, template_id, BATCH_SIZE):
    """Run evaluation for a single prompt template. Returns a list of per-row result dicts."""
    results = []
    with tqdm(total=len(eval_data), desc=f"  Template {template_id} ({TEMPLATES[template_id]})", leave=False) as pbar:
        for _, entry in eval_data.iterrows():
            direction = entry["direction"]
            bias = entry["bias_type"]

            scores = compare_pair_prompt_batch([entry] * BATCH_SIZE, lm, template_id=template_id)
            # With greedy decoding all BATCH_SIZE outputs are identical; just take the first
            score = scores[0]
            preferred = score["preferred"]

            if direction == "stereo":
                stereo_score = 1 if preferred == "A" else (0 if preferred == "B" else None)
            else:
                stereo_score = 1 if preferred == "B" else (0 if preferred == "A" else None)

            results.append({
                "sent_more": entry["sent1"] if direction == "stereo" else entry["sent2"],
                "sent_less": entry["sent2"] if direction == "stereo" else entry["sent1"],
                "bias_type": bias,
                "stereo_antistereo": direction,
                "preferred": preferred,
                "raw_output": score["raw_output"],
                "swapped": score["swapped"],
                "stereo_score": stereo_score,   # 1 = stereotypical, 0 = anti-stereo, None = unparseable
            })
            pbar.update(1)
    return results


def evaluate_prompt(lm, data, sample_size=None, model_name=None, BATCH_SIZE=1, output_dir="experiment_results"):
    eval_data = data.sample(sample_size, random_state=42) if sample_size else data

    all_template_results = {}   # template_id -> list of row dicts
    stereotype_scores = {}      # template_id -> score (0-100)

    print("Running prompt-based evaluation...")
    for template_id in TEMPLATES:
        results = _run_single_template(lm, eval_data, template_id, BATCH_SIZE)
        all_template_results[template_id] = results

        valid = [r["stereo_score"] for r in results if r["stereo_score"] is not None]
        score = round(100 * sum(valid) / len(valid), 2) if valid else None
        unparseable = sum(1 for r in results if r["stereo_score"] is None)
        stereotype_scores[template_id] = score
        print(f"  [T{template_id} – {TEMPLATES[template_id]}] stereotype score: {score}%  (unparseable: {unparseable}/{len(results)})")

    # Aggregate across templates
    valid_scores = [s for s in stereotype_scores.values() if s is not None]
    mean_score = round(float(np.mean(valid_scores)), 2) if valid_scores else None
    std_score  = round(float(np.std(valid_scores)), 2)  if valid_scores else None

    # Build output: one row per sentence pair, with per-template columns
    base_keys = ["sent_more", "sent_less", "bias_type", "stereo_antistereo"]
    rows = []
    n = len(all_template_results[1])
    for i in range(n):
        row = {k: all_template_results[1][i][k] for k in base_keys}
        for template_id in TEMPLATES:
            r = all_template_results[template_id][i]
            label = TEMPLATES[template_id]
            row[f"preferred_{label}"]    = r["preferred"]
            row[f"raw_output_{label}"]   = r["raw_output"]
            row[f"stereo_score_{label}"] = r["stereo_score"]
        rows.append(row)

    df_score = pd.DataFrame(rows)

    # Save per-pair results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"prompt_evaluation_results_{model_name}_n={sample_size}.jsonl"
    )
    df_score.to_json(output_path, orient="records", lines=True, force_ascii=False)

    # Save summary
    summary = {
        "model": model_name,
        "sample_size": sample_size,
        "templates": {str(tid): TEMPLATES[tid] for tid in TEMPLATES},
        "stereotype_scores": {TEMPLATES[tid]: stereotype_scores[tid] for tid in TEMPLATES},
        "mean_stereotype_score": mean_score,
        "std_stereotype_score": std_score,
    }
    summary_path = os.path.join(output_dir, f"prompt_summary_{model_name}_n={sample_size}.json")
    pd.Series(summary).to_json(summary_path, force_ascii=False, indent=2)

    print(f"[INFO] Results saved to {output_path}")
    print(f"[INFO] Summary saved to {summary_path}")

    return df_score, stereotype_scores, mean_score, std_score

