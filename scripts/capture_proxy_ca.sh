#!/usr/bin/env bash
# Save the root certificate that the network's HTTPS-inspecting proxy presents, combined with the
# system CA bundle, into certs/proxy_ca_bundle.pem. Verification stays ON afterwards, but only the
# proxy's CA (plus normal public CAs) is trusted. Trust-on-first-use: run it on the campus network.
#   bash scripts/capture_proxy_ca.sh
#   CA_BUNDLE=certs/proxy_ca_bundle.pem bash scripts/start_remote.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p certs
command -v openssl >/dev/null || { echo "openssl not found"; exit 1; }
CHAIN="$(openssl s_client -showcerts -servername pypi.org -connect pypi.org:443 </dev/null 2>/dev/null)"
# every PEM block in the presented chain; the last one is the proxy's (self-signed) root/intermediate
echo "$CHAIN" | awk '/BEGIN CERTIFICATE/{c++; inb=1} inb{f=("certs/chain_" c ".pem"); print > f} /END CERTIFICATE/{inb=0; close(f)}'
N="$(ls certs/chain_*.pem 2>/dev/null | wc -l)"
[[ "$N" -gt 0 ]] || { echo "could not read the certificate chain from pypi.org"; exit 1; }
echo "== certificate chain presented for pypi.org ($N certs):"
for f in $(ls certs/chain_*.pem | sort -V); do
  openssl x509 -in "$f" -noout -subject -issuer | sed 's/^/   /'
done
SYS=/etc/pki/tls/certs/ca-bundle.crt
[[ -f "$SYS" ]] || SYS=/etc/ssl/certs/ca-certificates.crt
{ [[ -f "$SYS" ]] && cat "$SYS"; cat certs/chain_*.pem; } > certs/proxy_ca_bundle.pem
rm -f certs/chain_*.pem
echo
echo "== wrote certs/proxy_ca_bundle.pem (system CAs + the chain above)"
echo "Check the subjects/issuers above look like your institution's proxy, then run:"
echo "  CA_BUNDLE=$ROOT/certs/proxy_ca_bundle.pem bash scripts/start_remote.sh"
