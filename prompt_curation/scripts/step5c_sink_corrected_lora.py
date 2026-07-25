import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, '.')
from step5_sink_corrected_metric import get_structural_indices, compute_corrected_metrics

def run(args):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    with open(args.attention_json) as f:
        data = json.load(f)

    results, summary = {}, {}
    for tmpl, regions in data["per_template"].items():
        raw_list, corr_list = [], []
        tmpl_results = []
        for region in tqdm(regions, desc=f"Template {tmpl}"):
            weights = region.get("att_weights", [])
            n = region.get("n_input_tokens", len(weights))
            prompt = region.get("prompt", "")
            li = region.get("label_token_indices", [])
            si = region.get("style_token_indices", [])
            if not weights or not prompt:
                continue
            struct_idx = get_structural_indices(tokenizer, prompt, "mistral")
            lr, sr, lc, sc, sm = compute_corrected_metrics(weights, li, si, struct_idx, n)
            tmpl_results.append({
                "image_id": region["image_id"], "mask_index": region["mask_index"],
                "label_mass_raw": lr, "label_mass_corrected": lc,
                "style_mass_raw": sr, "style_mass_corrected": sc,
            })
            raw_list.append(lr); corr_list.append(lc)
        results[tmpl] = tmpl_results
        factor = np.mean(corr_list) / max(np.mean(raw_list), 1e-10)
        summary[tmpl] = {
            "label_mass_raw": round(float(np.mean(raw_list)), 4),
            "label_mass_corrected": round(float(np.mean(corr_list)), 4),
            "improvement_factor": round(float(factor), 2),
            "n": len(tmpl_results),
        }
        print(f"  {tmpl}: raw={np.mean(raw_list)*100:.3f}%  corrected={np.mean(corr_list)*100:.3f}%  factor={factor:.2f}x")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attention_json", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
