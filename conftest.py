import os, time, socket, signal, subprocess, yaml, pathlib
import psutil
import pytest
from urllib.request import urlopen
import logging

def _find_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def _wait_http_ready(url: str, timeout_s: int = 300):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False

def _kill_process_tree(p: subprocess.Popen, wait=10):
    try:
        proc = psutil.Process(p.pid)
    except psutil.NoSuchProcess:
        return
    # Gentle first
    for c in proc.children(recursive=True):
        try: c.terminate()
        except Exception: pass
    try: proc.terminate()
    except Exception: pass
    _, alive = psutil.wait_procs([proc], timeout=wait)
    for a in alive:
        try: a.kill()
        except Exception: pass

def pytest_addoption(parser):
    group = parser.getgroup("serve")
    group.addoption("--models-yaml", default="models.yaml", help="Path to models.yaml")
    group.addoption("--vllm-bin", default="python -m vllm.entrypoints.openai.api_server",
                    help="vLLM server entry (e.g. 'python -m vllm.entrypoints.openai.api_server')")
    group.addoption("--sgl-bin", default="python -m sglang.launch_server",
                    help="sglang server entrypoint.")
    group.addoption("--host", default="127.0.0.1")
    group.addoption("--case-timeout", type=int, default=1200, help="Per-case timeout seconds")
    group.addoption("--gpu", default=os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"),
                    help="Visible GPU ids for this worker/case (optional)")
    group.addoption("--extra-serv-args", default="", help="Extra args passed to vLLM")
    group.addoption("--health-path", default="/v1/models", help="Health check path")

@pytest.fixture(scope="session")
def model_matrix(pytestconfig):
    cfg_path = pathlib.Path(pytestconfig.getoption("--models-yaml"))
    data = yaml.safe_load(cfg_path.read_text())
    return data["models"]

@pytest.fixture(scope="function")
def bmk_server(tmp_path, pytestconfig, case):
    """Spin up one vLLM server for this test, then teardown."""
    model_path = case["path"]
    backend = case["backend"]
    if not pathlib.Path(model_path).exists():
        pytest.skip(f"Model path not found: {model_path}")

    host = pytestconfig.getoption("--host")
    port = _find_free_port()
    vllm_bin = pytestconfig.getoption("--vllm-bin")
    sglang_bin = pytestconfig.getoption("--sgl-bin")
    extra = pytestconfig.getoption("--extra-serv-args")
    gpu_env = pytestconfig.getoption("--gpu")
    timeout_case = int(pytestconfig.getoption("--case-timeout"))
    health_path = pytestconfig.getoption("--health-path")

    logs = tmp_path / f"{case['name']}.server.log"
    endpoint = f"http://{host}:{port}"

    # Build command: default to OpenAI-compatible api_server
    cmd = []
    if backend == "vllm":
        cmd = vllm_bin.split() + [
            "--model", model_path,
            "--host", host,
            "--port", str(port),
        ]
    elif backend == "sglang":
        cmd = sglang_bin.split() + [
            "--model", model_path,
            "--host", host,
            "--port", str(port),
        ]
    else:
        raise RuntimeError("Not implemented yet...")
    if case.get("params"):
        cmd += list(case["params"])
    if extra:
        cmd += extra.split()

    env = os.environ.copy()
    if gpu_env:
        env["ROCR_VISIBLE_DEVICES"] = gpu_env
        env["CUDA_VISIBLE_DEVICES"] = gpu_env
    
    #TODO(billishyahao): only for vllm 
    if backend == "vllm":
        env.update({
            "VLLM_USE_V1": "1",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MHA": "0",
            "VLLM_V1_USE_PREFILL_DECODE_ATTENTION": "1",
            "SAFETENSORS_FAST_GPU": "1",
        })
    elif backend == "sglang":
        env.update({
            "SGLANG_USE_AITER": "1"
        })

    p = subprocess.Popen(
        cmd,
        stdout=open(logs, "wb"),
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    ok = _wait_http_ready(f"{endpoint}{health_path}", timeout_s=300)
    if not ok:
        _kill_process_tree(p)
        pytest.fail(f"vLLM server not ready in time. See logs: {logs}")

    logging.info(f"Server for {case['name']} is up...")
    start = time.time()
    def guard():
        if time.time() - start > timeout_case:
            raise TimeoutError(f"Case timeout {timeout_case}s for {case['name']}")
    try:
        yield {"endpoint": endpoint, "logs": logs, "guard": guard}
    finally:
        _kill_process_tree(p)

    logging.info(f"Teardown: Server for {case['name']} is killed...")