import json
import subprocess
import tempfile
import pathlib
import logging
from typing import Dict, Tuple, List, Optional


def _find_results_json(outdir: pathlib.Path) -> pathlib.Path:
    for p in sorted(outdir.glob("results*.json")):
        return p
    for p in outdir.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "results" in data:
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"No results JSON found under {outdir}")


def run_lm_eval(
    endpoint: str,
    model_path: str,
    tasks: List[str],
    num_fewshot: int = 5,
    limit: Optional[int] = 250,
    batch_size: str = "auto",
    num_concurrent: int = 256,
    max_retries: int = 10,
    max_gen_toks: int = 2048,
    apply_chat_template: bool = True,
    extra_model_args: Optional[Dict[str, str]] = None,
    lm_eval_bin: str = "lm_eval",
    output_path: Optional[pathlib.Path] = None,
) -> Tuple[dict, dict]:

    base_url = f"{endpoint}/v1/completions"

    model_args = {
        "model": model_path,
        "base_url": base_url,
        "num_concurrent": str(num_concurrent),
        "max_retries": str(max_retries),
        "max_gen_toks": str(max_gen_toks),
    }
    if extra_model_args:
        model_args.update({k: str(v) for k, v in extra_model_args.items()})

    model_args_kv = ",".join([f"{k}={v}" for k, v in model_args.items()])


    cmd = [
        lm_eval_bin,
        "--model", "local-completions",
        "--model_args", model_args_kv,
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if apply_chat_template:
        cmd += ["--apply_chat_template"]


    logging.info(f"Start test...")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"lm_eval failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    # Parse results.json

    logging.info(f"Parsing result...")
    results_file = _find_results_json(output_path)
    data = json.loads(results_file.read_text(encoding="utf-8"))

    # Get primary metric for first task
    # TODO(billishyahao): to support multi task
    first_task = tasks[0]
    results = data["results"]

    return results[first_task], results
