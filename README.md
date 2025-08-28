# PyServeEval

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/tested%20with-pytest-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PyServeEval** is a lightweight, pytest-based framework for evaluating **LLM serving** backends.  
It currently supports **vLLM** and **SGLang**, and integrates with **lm_eval** for benchmark accuracy testing.  

---

## ✨ Features

- **Backends**
  - ✅ vLLM (with environment variables like `VLLM_USE_V1`, `VLLM_ROCM_USE_AITER`, …)  
  - ✅ SGLang (`sglang.launch_server`)  

- **Accuracy Evaluation**
  - Integrates with [`lm_eval`](https://github.com/EleutherAI/lm-evaluation-harness)  
  - Example task: **gsm8k** with few-shot settings  
  - JSON results automatically stored under pytest’s tmp directory  

- **Performance Benchmarks**
  - Can be extended to wrap `sglang.bench_serving` or other throughput/latency tools  

- **Pytest-native**
  - Use fixtures like `vllm_server` / `sglang_server` to manage lifecycle  
  - Results written into `tmp_path` (e.g. `/tmp/pytest-of-root/.../results.json`)  

---

## 🚀 Quickstart

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run accuracy test 
```bash
pytest
```

This will:
1. Launch a LLM server with configured env vars  
2. Call `lm_eval` with task `gsm8k`  
3. Save results JSON to pytest tmp directory  

Example result path:
```
/tmp/pytest-of-root/pytest-current/test_lm_eval_gsm8k_Llama_3_3_70/lm_eval_result.json
```

Enable more info loggging through `--log-cli-level INFO`


### Run accuracy test on VLLM or SGL seperately
```bash
pytest -v --reruns 3 # run all testcases with retry
pytest -v --log-cli-level INFO -k "vllm"  # vllm
pytest -v --log-cli-level INFO -k "sgl"   # sglang
```

---

## ⚙️ Configuration

Model settings live in `models.yaml`.  
Example:

```yaml
models:
  - name: Llama-3.1-405B-Instruct
    path: /models/Llama-3.1-405B-Instruct-MXFP4-Preview
    port: 30000
```

Eval settings live in `lm_eval_config.yaml`  
Example:

```yaml
lm_eval:
  extra_model_args:
    base_url: "http://127.0.0.1:<port>/v1/chat/completions"
```


---

## 📂 Project Structure

```
py-serve-eval/
│── pytest.ini
│── requirements.txt
│── conftest.py             # fixture instance
│── README.md
│
│── models.yaml             # model test case definition
│── lm_eval_config.yaml     # eval  config
│
│── tests/
│   ├── test_lm_eval.py     # accuracy eval via lm_eval
│── lm_eval_runner.py       # lm eval runner
```

---


## 📜 License
[MIT](LICENSE)


---

## 🛣️ Roadmap

- [x] Initial release with vLLM backend
- [x] Support for lm-eval tasks (e.g., GSM8K)
- [x] SGLang backend integration
- [x] Pure Python integration
- [ ] Parallel execution with pytest-xdist (smarter port assignment, model-aware sharding)
- [ ] Native Slurm & Kubernetes runners
- [ ] More benchmark tasks (MMLU, BBH, HumanEval)
- [ ] Performance dashboard (latency, throughput, invalids)
- [ ] CI/CD integration with GitHub Actions
