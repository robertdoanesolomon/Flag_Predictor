#!/usr/bin/env python
"""
One-off set-up for flag alerts, run by .github/workflows/deploy_push.yml:

    python push/setup_push.py namespace   # create the KV namespace if needed, print its id
    python push/setup_push.py keys        # create the VAPID key pair in KV if needed
    python push/setup_push.py config      # write docs/preview/push_config.json (Worker address, public key)

Needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID. The VAPID private key never leaves
Cloudflare KV and the GitHub job that sends the alerts.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloudflare import Cloudflare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'docs' / 'preview' / 'push_config.json'


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b'=').decode()


def make_keys() -> tuple[str, str]:
    """(private key PEM, public key as base64url uncompressed point) for VAPID (P-256)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return pem, b64url(pub)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    cf = Cloudflare()
    if not cf.configured:
        sys.exit('CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID are not set')
    if cmd == 'namespace':
        print(cf.ensure_namespace())
    elif cmd == 'keys':
        if not cf.find_namespace():
            sys.exit('no KV namespace yet: run "namespace" first')
        if cf.get('config:vapid_public') and cf.get('config:vapid_private'):
            print('VAPID keys already set')
            return
        pem, pub = make_keys()
        cf.put('config:vapid_private', pem)
        cf.put('config:vapid_public', pub)
        print('VAPID keys created')
    elif cmd == 'config':
        if not cf.find_namespace():
            sys.exit('no KV namespace yet')
        url = os.environ.get('WORKER_URL') or None
        if not url:
            try:
                url = cf.worker_url()
            except Exception as exc:       # token without the account-settings permission
                print(f'could not look up the workers.dev address: {exc}')
        pub = cf.get('config:vapid_public')
        if not url or not pub:
            sys.exit(f'missing worker address ({url}) or public key ({bool(pub)}): '
                     'open Workers & Pages in the Cloudflare dashboard once to choose a workers.dev subdomain')
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps({'worker': url, 'vapid_public': pub}, indent=1) + '\n')
        print(f'wrote {CONFIG.relative_to(ROOT)}: {url}')
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
