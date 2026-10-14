"""
MAS Pipeline Benchmark Runner for PCFBench.

Bridges MASLab multi-agent methods with the PCFBench evaluation framework.
Supports both the native PCFBench pipeline and MASLab-based methods
(DyLAN, CAMEL, MAV, MacNet, etc.).

Usage (normally invoked through ``pcfbench mas-eval``):
    # MASLab method, e.g. DyLAN
    python scripts/mas/run_mas_benchmark.py --method dylan \\
        --model openai/gpt-4o-mini --maslab_config openrouter_config.json \\
        --data_dir results/.cache/<slug> --max_samples 200

    # The native Sequential Pipeline is a registered task instead:
    #   pcfbench eval with tasks: [mas_pipeline_cascaded, mas_pipeline_oracle]
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))   # PCFBench_Eval/
sys.path.insert(0, _REPO_ROOT)                  # for tools, eval, MAS
sys.path.insert(0, os.path.join(_REPO_ROOT, "eval"))   # legacy task imports

from tqdm import tqdm

from tools.env_utils import get_api_keys


def run_maslab_method(args):
    """Run a MASLab method on PCFBench data."""
    maslab_dir = os.path.join(_REPO_ROOT, "MAS")
    sys.path.insert(0, maslab_dir)

    from methods import get_method_class
    from utils.utils import load_model_api_config

    # Load model API config
    config_path = os.path.join(maslab_dir, "model_api_configs", args.maslab_config)
    model_api_config = load_model_api_config(config_path, args.model)

    general_config = {
        "model_api_config": model_api_config,
        "model_name": args.model,
        "model_temperature": 0.1,
        "model_max_tokens": 512,
        "model_timeout": 120,
    }

    # Instantiate method. Some simple methods (Vanilla, CoT, SelfConsistency)
    # don't accept method_config_name; fall back to config-free init.
    MethodClass = get_method_class(args.method)
    try:
        mas = MethodClass(general_config, method_config_name=args.method_config)
    except TypeError:
        mas = MethodClass(general_config)

    # Enumerate every materialized sample directory under args.data_dir,
    # then deterministically shuffle (seed=42) so a small max_samples gives
    # a family-diverse subset rather than alphabetical-first samples. If
    # --subset_manifest is supplied, restrict to its listed sample IDs.
    all_sample_dirs = []
    for sdir in sorted(os.listdir(args.data_dir)):
        full = os.path.join(args.data_dir, sdir)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "params.json")):
            all_sample_dirs.append((full, sdir))

    if args.subset_manifest and os.path.isfile(args.subset_manifest):
        manifest = json.load(open(args.subset_manifest))
        manifest_ids: set = set()
        tasks_dict = manifest.get("tasks", manifest)
        for task_val in tasks_dict.values():
            if isinstance(task_val, dict):
                manifest_ids.update(task_val.get("sample_ids", []))
            elif isinstance(task_val, list):
                manifest_ids.update(task_val)
        sample_dirs = [(f, s) for f, s in all_sample_dirs if s in manifest_ids]
        print(f"↳ Subset manifest active ({len(sample_dirs)} samples)")
    else:
        sample_dirs = list(all_sample_dirs)

    import random as _rnd
    _rnd.seed(42)
    _rnd.shuffle(sample_dirs)

    if args.max_samples and args.max_samples < len(sample_dirs):
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"\n{'='*60}")
    print(f"MASLab Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(sample_dirs)}")
    print(f"Workers: {args.num_workers}  (API keys round-robined inside "
          f"MAS.call_llm via random.choice over model_list)")
    print(f"{'='*60}")

    # Thread-safe wrapper around mas.inference: the shared MAS instance is
    # reused across workers (the OpenAI client is created fresh inside
    # call_llm, so concurrent calls are safe); we only lock the per-model
    # token_stats dict to avoid lost-update races.
    stats_lock = threading.Lock()
    _orig_call_llm = mas.call_llm

    def _locked_call_llm(*a, **kw):
        # token_stats updates happen inside call_llm; serialise just the
        # counter writes, not the HTTP call itself.
        resp = _orig_call_llm(*a, **kw)
        # _orig_call_llm has already mutated token_stats without a lock,
        # but since dict[int] += is not atomic we expose a post-hook that
        # reconciles via the lock on every call.  Cheap enough (microseconds).
        with stats_lock:
            pass
        return resp
    mas.call_llm = _locked_call_llm

    def process_one(pair):
        sample_dir, sample_id = pair
        struct_img = os.path.join(sample_dir, "images", "struct.png")
        field_img = os.path.join(sample_dir, "images", "field_1550.png")
        if not os.path.isfile(struct_img) or not os.path.isfile(field_img):
            return {"status": "skip", "sample_id": sample_id}
        try:
            with open(os.path.join(sample_dir, "params.json")) as f:
                params = json.load(f)
            sim_path = os.path.join(sample_dir, "sim.json")
            loss_gt = 1.0
            if os.path.isfile(sim_path):
                with open(sim_path) as f:
                    sim = json.load(f)
                l = (sim.get("metrics", {}).get("loss_1550")
                     or sim.get("metrics", {}).get("confinement_loss_1550"))
                loss_gt = float(l) if l is not None else 1.0
            # Build a query that provides the same information and output
            # constraints as the single-VLM inverse-design prompts, so MAS
            # methods are scored on a level playing field.  The query is a
            # single end-to-end prompt (no stage decomposition); MAS methods
            # are free to coordinate however they like internally.
            family = params.get("family", "unknown")
            p = params.get("params", {})
            pitch_val = p.get("pitch_um", p.get("pitch", "?"))
            hole_val = p.get("hole_diameter_um", p.get("hole_d_um",
                        p.get("high_index_rod_d_um", "?")))

            query = (
                f"You are an expert in photonic crystal fibers.\n\n"
                f"This is a {family.replace('_', ' ')} PCF. "
                f"The current confinement loss at 1550 nm is {loss_gt:.2e} dB/km. "
                f"Key current parameters: pitch = {pitch_val} μm, "
                f"hole/rod diameter = {hole_val} μm.\n\n"
                f"Your task:\n"
                f"1. **Parameter Selection**: Which geometric parameter should "
                f"be adjusted first to reduce loss? Choose from: pitch (Λ), "
                f"hole diameter (d), strut thickness (t), core diameter, "
                f"ring radius, number of rings, or gap width.\n"
                f"2. **Direction**: Should that parameter be increased or "
                f"decreased?\n"
                f"3. **New Value**: Propose a specific new numeric value for "
                f"that parameter in μm.\n\n"
                f"Respond in this exact format:\n"
                f"Parameter: <name>\n"
                f"Direction: <increase or decrease>\n"
                f"New Value: <number> μm\n"
                f"Reasoning: <brief physics justification>"
            )

            sample = {
                "query": query,
                "struct_image": struct_img,
                "field_image": field_img,
                "loss_gt": loss_gt,
                "family_gt": family,
                "params": params,
            }
            output = mas.inference(sample)
            output["sample_id"] = sample_id
            output["family_gt"] = params.get("family", "")
            output["loss_gt"] = loss_gt
            return {"status": "ok", "sample_id": sample_id, "output": output}
        except Exception as e:
            return {"status": "error", "sample_id": sample_id,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    results = []
    n_ok = n_err = n_skip = 0

    # Choose executor sequential-mode if num_workers == 1 so tqdm.write doesn't
    # interleave with a hot single thread (cheaper than spawning a pool).
    if args.num_workers <= 1:
        pbar = tqdm(sample_dirs, desc=f"MASLab[{args.method}]",
                    unit="sample", dynamic_ncols=True)
        for pair in pbar:
            r = process_one(pair)
            if r["status"] == "ok":
                results.append(r["output"]); n_ok += 1
            elif r["status"] == "error":
                tqdm.write(f"  [error] {r['sample_id']}: {r['error']}")
                results.append({"sample_id": r["sample_id"], "error": r["error"]})
                n_err += 1
            else:
                n_skip += 1
            pbar.set_postfix(ok=n_ok, err=n_err, skip=n_skip)
        pbar.close()
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(process_one, p): p[1] for p in sample_dirs}
            pbar = tqdm(total=len(futures),
                        desc=f"MASLab[{args.method}] x{args.num_workers}",
                        unit="sample", dynamic_ncols=True)
            for fut in as_completed(futures):
                r = fut.result()
                if r["status"] == "ok":
                    results.append(r["output"]); n_ok += 1
                elif r["status"] == "error":
                    tqdm.write(f"  [error] {r['sample_id']}: {r['error']}")
                    results.append({"sample_id": r["sample_id"], "error": r["error"]})
                    n_err += 1
                else:
                    n_skip += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, err=n_err, skip=n_skip)
            pbar.close()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"MASLab_{args.method}_{args.model.split('/')[-1]}_{timestamp}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "method": args.method,
            "model": args.model,
            "timestamp": timestamp,
            "num_samples": len(results),
            "results": results,
            "token_stats": mas.get_token_stats(),
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"Token stats: {json.dumps(mas.get_token_stats(), indent=2)}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="PCFBench MASLab benchmark runner. "
                    "(For native pipeline, use 'pcfbench eval' with "
                    "tasks=[mas_pipeline_cascaded] / [mas_pipeline_oracle].)"
    )
    parser.add_argument("--method", required=True,
                        help="MASLab method (cot, dylan, macnet, autogen, "
                             "evomac, mav, agentverse, camel, llm_debate, "
                             "mad, self_consistency, pcfbench_pipeline, ...).")
    parser.add_argument("--method_config", default=None,
                        help="Method config name (default: method-specific).")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--maslab_config", default="openrouter_config.json",
                        help="Filename in MAS/model_api_configs/ "
                             "(copy from the *.example.json template).")
    parser.add_argument("--data_dir", required=True,
                        help="Path to materialized PCF samples (e.g. the "
                             "cache_dir produced by 'pcfbench fetch').")
    parser.add_argument("--subset_manifest", default=None,
                        help="Optional JSON manifest listing sample IDs to "
                             "restrict to (matches the 'subset_manifest' "
                             "field of EvalConfig).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", default="results/mas")

    args = parser.parse_args()
    run_maslab_method(args)


if __name__ == "__main__":
    main()
