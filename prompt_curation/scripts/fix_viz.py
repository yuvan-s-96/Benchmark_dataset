import json
import numpy as np

with open("../results/template_comparison_979.json") as f:
    d = json.load(f)

# Build summary
summary = {}
for tmpl, regions in d["per_template"].items():
    masses  = np.array([r["label_attention_mass"] for r in regions])
    styles  = np.array([r.get("style_attention_mass", 0) for r in regions])
    refusals = sum(1 for r in regions
                   if any(p in r.get("instruction","").lower()
                          for p in ["i'm an ai","i cannot","language model","as an ai"]))
    summary[tmpl] = {
        "label_mass": {
            "mean":   round(float(masses.mean()), 6),
            "median": round(float(np.median(masses)), 6),
            "min":    round(float(masses.min()), 6),
            "max":    round(float(masses.max()), 6),
        },
        "style_mass": {"mean": round(float(styles.mean()), 6)},
        "refusal_rate": round(refusals / len(regions) * 100, 1),
        "n": len(regions),
    }

# Inject summary back into JSON
d["summary"] = summary
with open("../results/template_comparison_979.json", "w") as f:
    json.dump(d, f, indent=2)
print("Summary injected into template_comparison_979.json")
for t, s in sorted(summary.items(), key=lambda x: x[1]["label_mass"]["mean"], reverse=True):
    print(f"  {t}: label={s['label_mass']['mean']*100:.3f}% "
          f"style={s['style_mass']['mean']*100:.3f}% "
          f"refusal={s['refusal_rate']:.1f}%")
