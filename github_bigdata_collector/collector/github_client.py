import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import requests


class GitHubClient:
    """Small GitHub REST client (v3) with token auth + basic pagination."""

    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError(
                "Missing GITHUB_TOKEN environment variable. "
                "Set it to a GitHub Personal Access Token (classic or fine-grained)."
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

    def _maybe_sleep_rate_limit(self, resp: requests.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is None or reset is None:
            return
        try:
            remaining_i = int(remaining)
            reset_i = int(reset)
        except Exception:
            return

        if remaining_i <= 0:
            wait_s = max(0, reset_i - int(time.time())) + 2
            print(f"[rate-limit] Sleeping {wait_s}s until reset...")
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

                # Rate limit / forbidden
                if resp.status_code == 403:
                    self._maybe_sleep_rate_limit(resp)
                    if attempt < max_retries - 1:
                        time.sleep(min(2**attempt, 30) + random.random())
                        continue

                # Transient errors
                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(min(2**attempt, 30) + random.random())
                    continue

                if resp.status_code >= 400:
                    raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text[:800]}")

                self._maybe_sleep_rate_limit(resp)
                return resp.json()

            except Exception as e:
                last_err = e
                time.sleep(min(2**attempt, 30) + random.random())

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
        params = dict(params or {})
        params["per_page"] = per_page

        page = 1
        pages = 0
        while True:
            params["page"] = page
            data = self.get(path, params=params)

            if not isinstance(data, list):
                # Some endpoints return dict; treat as a single page.
                yield [data]
                return

            if not data:
                return

            yield data

            page += 1
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return
import os
import time
import random
import requests
from typing import Any, Dict, Optional, Iterable, List


class GitHubClient:
    """
    GitHub REST client with:
    - Token auth
    - Retries + exponential backoff
    - Rate-limit sleep handling
    - Simple pagination helpers
    """

    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("Missing GITHUB_TOKEN environment variable.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-bigdata-collector"
        })

    def _maybe_sleep_rate_limit(self, resp: requests.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is None or reset is None:
            return
        try:
            remaining_i = int(remaining)
            reset_i = int(reset)
        except Exception:
            return

        if remaining_i <= 0:
            wait_s = max(0, reset_i - int(time.time())) + 2
            print(f"[rate-limit] Sleeping {wait_s}s until reset...")
            time.sleep(wait_s)

    def request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 7) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_err = None

        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, params=params, timeout=40)

                # Rate limit or forbidden
                if resp.status_code == 403:
                    self._maybe_sleep_rate_limit(resp)
                    if attempt < max_retries - 1:
                        time.sleep(min(2 ** attempt, 30) + random.random())
                        continue

                # Transient errors
                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 30) + random.random())
                    continue

                if resp.status_code >= 400:
                    raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text[:800]}")

                self._maybe_sleep_rate_limit(resp)
                return resp.json()

            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30) + random.random())

        raise RuntimeError(f"Failed after retries: {method} {url}. Last error: {last_err}")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self.request("GET", path, params=params)

    def paginate(self, path: str, params: Optional[Dict[str, Any]] = None,
                 per_page: int = 100, max_pages: Optional[int] = None) -> Iterable[List[dict]]:
        """
        Pagination using ?page= and ?per_page=
        Returns pages (list of dict items).
        """
        params = dict(params or {})
        params["per_page"] = per_page

        page = 1
        pages = 0
        while True:
            params["page"] = page
            data = self.get(path, params=params)

            if not isinstance(data, list):
                # If endpoint returns dict, treat as one page
                yield [data]
                return

            if not data:
                return

            yield data

            page += 1
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return
