"""Minimal Cloudflare API client for the alerts KV namespace (needs CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID)."""

from __future__ import annotations

import os
from typing import Iterator, Optional

import requests

API = 'https://api.cloudflare.com/client/v4'
NAMESPACE_TITLE = 'flagpredictor-alerts'
WORKER_NAME = 'flagpredictor-alerts'


class Cloudflare:
    def __init__(self, token: Optional[str] = None, account: Optional[str] = None):
        self.token = token or os.environ.get('CLOUDFLARE_API_TOKEN', '')
        self.account = account or os.environ.get('CLOUDFLARE_ACCOUNT_ID', '')
        self.s = requests.Session()
        self.s.headers['Authorization'] = f'Bearer {self.token}'
        self.ns: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.account)

    def _url(self, path: str) -> str:
        return f'{API}/accounts/{self.account}{path}'

    def _json(self, r: requests.Response) -> dict:
        try:
            j = r.json()
        except ValueError:
            r.raise_for_status()
            raise
        if not j.get('success', r.ok):
            raise RuntimeError(f'Cloudflare API {r.request.method} {r.url.split("/accounts/")[-1]}: {j.get("errors")}')
        return j

    # namespace
    def find_namespace(self) -> Optional[str]:
        page = 1
        while True:
            j = self._json(self.s.get(self._url('/storage/kv/namespaces'), params={'per_page': 100, 'page': page}))
            for ns in j['result']:
                if ns['title'] == NAMESPACE_TITLE:
                    self.ns = ns['id']
                    return self.ns
            if len(j['result']) < 100:
                return None
            page += 1

    def ensure_namespace(self) -> str:
        if self.find_namespace():
            return self.ns
        j = self._json(self.s.post(self._url('/storage/kv/namespaces'), json={'title': NAMESPACE_TITLE}))
        self.ns = j['result']['id']
        return self.ns

    # values
    def get(self, key: str) -> Optional[str]:
        r = self.s.get(self._url(f'/storage/kv/namespaces/{self.ns}/values/{key}'))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text

    def put(self, key: str, value: str) -> None:
        self._json(self.s.put(self._url(f'/storage/kv/namespaces/{self.ns}/values/{key}'), data=value.encode(),
                              headers={'Content-Type': 'text/plain'}))

    def delete(self, key: str) -> None:
        r = self.s.delete(self._url(f'/storage/kv/namespaces/{self.ns}/values/{key}'))
        if r.status_code not in (200, 404):
            self._json(r)

    def keys(self, prefix: str) -> Iterator[str]:
        cursor = None
        while True:
            params = {'prefix': prefix, 'limit': 1000}
            if cursor:
                params['cursor'] = cursor
            j = self._json(self.s.get(self._url(f'/storage/kv/namespaces/{self.ns}/keys'), params=params))
            for k in j['result']:
                yield k['name']
            cursor = (j.get('result_info') or {}).get('cursor')
            if not cursor:
                return

    # workers.dev address of the alerts Worker
    def worker_url(self) -> Optional[str]:
        j = self._json(self.s.get(self._url('/workers/subdomain')))
        sub = (j.get('result') or {}).get('subdomain')
        return f'https://{WORKER_NAME}.{sub}.workers.dev' if sub else None
