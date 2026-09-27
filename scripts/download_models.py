"""Download (or verify) all model weights into $HF_HOME. Runs on the machine that trains.

Honours REQUESTS_CA_BUNDLE (set by start_remote.sh to the system CA bundle, which trusts an HTTPS-
intercepting campus proxy). ALLOW_INSECURE_DOWNLOADS=1 disables certificate checks as a last resort.
Exit code 1 if a required model could not be obtained.
"""
import os
import sys

# hf_xet (Rust downloader) has its own TLS trust store and ignores REQUESTS_CA_BUNDLE; use plain HTTPS.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests
from huggingface_hub import configure_http_backend, snapshot_download

REQUIRED = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"]          # full run
OPTIONAL = ["intfloat/multilingual-e5-small", "cross-encoder/ms-marco-MiniLM-L6-v2"]  # smoke/dev
KEEP = ["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "tokenizer*", "sentencepiece*"]
SKIP = ["onnx/*", "*.onnx", "*.pt", "*.h5", "*.msgpack", "*.ot", "imgs/*", "*.pth"]

if os.environ.get("ALLOW_INSECURE_DOWNLOADS") == "1":
    import urllib3
    urllib3.disable_warnings()

    def _insecure_session() -> requests.Session:
        s = requests.Session()
        s.verify = False
        return s
    configure_http_backend(backend_factory=_insecure_session)
    print("WARNING: TLS certificate verification disabled for model downloads")

failed = []
for m in REQUIRED + OPTIONAL:
    try:
        print("cached    ", m, "->", snapshot_download(m, local_files_only=True), flush=True)
        continue
    except Exception:
        pass
    try:
        print("downloading", m, flush=True)
        print("  ->", snapshot_download(m, allow_patterns=KEEP, ignore_patterns=SKIP), flush=True)
    except Exception as e:
        print(f"  FAILED {m}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        if m in REQUIRED:
            failed.append(m)
if failed:
    print("Missing required models:", failed)
    sys.exit(1)
print("ALL REQUIRED MODELS AVAILABLE")
