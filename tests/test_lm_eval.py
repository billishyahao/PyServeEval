import json, pathlib, pytest, yaml
from lm_eval_runner import run_lm_eval

LM_EVAL_CFG = yaml.safe_load(open("lm_eval_config.yaml", "r", encoding="utf-8"))
MODELS = yaml.safe_load(open("models.yaml", "r", encoding="utf-8"))["models"]

@pytest.mark.parametrize(
    "case",
    [pytest.param(m, id=m["name"]) for m in MODELS]
)
def test_lm_eval_gsm8k(bmk_server, case, tmp_path):
    bmk_server["guard"]()
    endpoint = bmk_server["endpoint"]
    threshold = float(case.get("expect").get("min"))
    metric = case.get("expect").get("metric")

    cfg = LM_EVAL_CFG.get("lm_eval", {})
    task0_result, result = run_lm_eval(
        endpoint=endpoint,
        model_path=case["path"],
        tasks=cfg.get("tasks", ["gsm8k"]),
        num_fewshot=int(cfg.get("num_fewshot", 5)),
        limit=cfg.get("limit", 250),
        batch_size=str(cfg.get("batch_size", "auto")),
        num_concurrent=int(cfg.get("num_concurrent", 256)),
        max_retries=int(cfg.get("max_retries", 10)),
        max_gen_toks=int(cfg.get("max_gen_toks", 2048)),
        apply_chat_template=bool(cfg.get("apply_chat_template", True)),
        extra_model_args=cfg.get("extra_model_args"),
        output_path=tmp_path
    )

    dest = tmp_path / "lm_eval_result.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    score = task0_result[metric]

    assert score >= threshold, f"{case['name']} lm_eval {cfg.get('tasks', ['gsm8k'])[0]} score {score:.3f} < {threshold:.3f}. See {dest}"
