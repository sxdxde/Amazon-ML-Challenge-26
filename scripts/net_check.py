"""Find a way for Python (pip / huggingface_hub) to reach pypi.org + huggingface.co over HTTPS.

Tries, in order: certifi (what pip and requests use by default), Python's default trust store, then
the candidate CA files given on the command line. Prints diagnostics to stderr and, on success, the
CA file to export on the LAST stdout line ("DEFAULT" = nothing to export). Exit 1 if nothing works.
"""
import ssl
import sys
import urllib.request

URLS = ["https://pypi.org/simple/pip/", "https://huggingface.co/api/models/BAAI/bge-m3"]


def works(ctx: ssl.SSLContext, label: str) -> bool:
    for u in URLS:
        try:
            urllib.request.urlopen(u, timeout=20, context=ctx).read(64)
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] {u} -> {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
            return False
    print(f"  [{label}] OK", file=sys.stderr)
    return True


def main() -> int:
    try:
        import certifi
        if works(ssl.create_default_context(cafile=certifi.where()), "certifi"):
            print("DEFAULT")
            return 0
    except ImportError:
        pass
    if works(ssl.create_default_context(), "python-default"):
        print(ssl.get_default_verify_paths().openssl_cafile or "DEFAULT")
        return 0
    for path in sys.argv[1:]:
        if not path:
            continue
        try:
            ctx = ssl.create_default_context(cafile=path)
        except Exception as e:  # noqa: BLE001
            print(f"  [{path}] unusable: {e}", file=sys.stderr)
            continue
        if works(ctx, path):
            print(path)
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
