from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


class GitHubClient:
    """
    GitHub REST client (v3) with:
    - Token auth
    - Retries + exponential backoff
    - Rate-limit sleep handling
    - Secondary rate-limit / abuse detection backoff
    - Pagination helpers
    """

    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError(
                "Missing GITHUB_TOKEN environment variable. "
                "Set it to a GitHub Personal Access Token."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "github-bigdata-collector",
            }
        )

    def _rate_limit_info(self, resp: requests.Response) -> Tuple[Optional[int], Optional[int]]:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        try:
            remaining_i = int(remaining) if remaining is not None else None
            reset_i = int(reset) if reset is not None else None
            return remaining_i, reset_i
        except Exception:
            return None, None

    def _maybe_sleep_rate_limit(self, resp: requests.Response) -> None:
        remaining_i, reset_i = self._rate_limit_info(resp)
        if remaining_i is None or reset_i is None:
            return
        if remaining_i <= 0:
            wait_s = max(0, reset_i - int(time.time())) + 2
            print(f"[rate-limit] Remaining=0. Sleeping {wait_s}s until reset...")
            time.sleep(wait_s)

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 7,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, params=params, timeout=40)

                # 429 Too Many Requests (sometimes happens)
                if resp.status_code == 429:
                    sleep_s = min(2**attempt, 60) + random.random()
                    print(f"[429] Too many requests. Sleeping {sleep_s:.2f}s...")
                    time.sleep(sleep_s)
                    continue

                # 403: could be rate-limit OR forbidden OR secondary-limit
                if resp.status_code == 403:
                    remaining_i, reset_i = self._rate_limit_info(resp)

                    # If remaining == 0, it's normal rate-limit
                    if remaining_i is not None and remaining_i <= 0 and reset_i is not None:
                        self._maybe_sleep_rate_limit(resp)
                        continue

                    # Otherwise likely "secondary rate limit" / abuse / forbidden
                    # Backoff a bit and retry a few times.
                    msg = (resp.text or "")[:300].replace("\n", " ")
                    sleep_s = min(2**attempt, 90) + 1 + random.random()
                    print(f"[403] Forbidden/secondary-limit? {msg} | sleeping {sleep_s:.2f}s...")
                    time.sleep(sleep_s)
                    continue

                # Transient server errors
                if resp.status_code in (500, 502, 503, 504):
                    sleep_s = min(2**attempt, 60) + random.random()
                    print(f"[{resp.status_code}] Transient error. Sleeping {sleep_s:.2f}s...")
                    time.sleep(sleep_s)
                    continue

                if resp.status_code >= 400:
                    raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text[:800]}")

                # Normal request: after success, still respect rate-limit headers
                self._maybe_sleep_rate_limit(resp)
                return resp.json()

            except Exception as e:
                last_err = e
                sleep_s = min(2**attempt, 60) + random.random()
                print(f"[retry] Error: {e}. Sleeping {sleep_s:.2f}s...")
                time.sleep(sleep_s)

        raise RuntimeError(f"Failed after retries: {method} {url}. Last error: {last_err}")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self.request("GET", path, params=params)

    def paginate(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        per_page: int = 100,
        max_pages: Optional[int] = None,
    ) -> Iterable[List[dict]]:
        """
        Pagination using ?page= and ?per_page=.
        Yields pages as lists of dict.
        """
        params = dict(params or {})
        params["per_page"] = per_page

        page = 1
        pages = 0
        while True:
            params["page"] = page
            data = self.get(path, params=params)

            if not isinstance(data, list):
                # Some endpoints return dict; treat it as one page.
                yield [data]
                return

            if not data:
                return

            yield data

            page += 1
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return