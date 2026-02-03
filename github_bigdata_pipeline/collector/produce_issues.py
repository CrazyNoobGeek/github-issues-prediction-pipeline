
"""
Kafka producer for GitHub issues.
Leverages existing GitHubClient and collector modules.
"""
from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, List
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError

# Import existing collector modules
from github_bigdata_pipeline.collector.github_client import GitHubClient
from github_bigdata_pipeline.collector.repos import collect_repos
from github_bigdata_pipeline.collector.issues import collect_issues_for_repo, max_updated_at, subtract_overlap
from github_bigdata_pipeline.collector.state import (
    load_json,
    save_json,
    repo_state_path,
    issues_state_path,
    get_since,
    set_since,
    set_repo_run_meta,
)
from github_bigdata_pipeline.collector.io_utils import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class GitHubKafkaProducer:
    """Kafka producer for streaming GitHub issues."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        github_client: GitHubClient
    ):
        self.topic = topic
        self.gh = github_client
        
        logger.info(f"Initializing Kafka producer for {bootstrap_servers}")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=5,
            max_in_flight_requests_per_connection=1,
            compression_type='gzip',
            batch_size=16384,
            linger_ms=10
        )
        
        logger.info(f"Kafka producer initialized for topic: {self.topic}")

    def publish_issue(self, issue: Dict[str, Any]) -> bool:
        """Publish a single issue to Kafka."""
        try:
            # Use issue ID as partition key
            key = str(issue.get('id', ''))
            
            future = self.producer.send(
                self.topic,
                key=key,
                value=issue
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            logger.debug(
                f"Published issue {issue.get('repo_full_name')}/{issue.get('number')} "
                f"to partition {record_metadata.partition} at offset {record_metadata.offset}"
            )
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error publishing issue {issue.get('number')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error publishing issue {issue.get('number')}: {e}")
            return False

    def publish_batch(self, issues: List[Dict[str, Any]]) -> int:
        """Publish a batch of issues to Kafka."""
        published = 0
        for issue in issues:
            if self.publish_issue(issue):
                published += 1
        
        # Flush to ensure all messages are sent
        self.producer.flush()
        return published

    def close(self):
        """Close the Kafka producer."""
        logger.info("Closing Kafka producer...")
        self.producer.close()
        logger.info("Kafka producer closed")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration file."""
    if not os.path.exists(path):
        logger.warning(f"Config file not found: {path}, using defaults")
        return {
            "state_dir": "data/state",
            "repo_sources": {
                "orgs": ["tensorflow"],
                "search_queries": ["stars:>10000 language:python"],
                "max_repos_total": 50
            },
            "issues_collection": {
                "default_since": "2024-01-01T00:00:00Z",
                "max_repos_per_run": 20,
                "max_pages_per_repo": 10,
                "include_state": "all",
                "since_overlap_seconds": 600,
                "collect_first_comment": False,
                "max_first_comment_fetch": 0
            }
        }
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_collection_cycle(
    producer: GitHubKafkaProducer,
    config: Dict[str, Any]
) -> Dict[str, int]:
    """
    Run one complete collection cycle:
    1. Discover repositories
    2. Collect issues from each repository
    3. Publish issues to Kafka
    """
    stats = {
        'repos_collected': 0,
        'issues_collected': 0,
        'issues_published': 0,
        'errors': 0
    }

    state_dir = config.get("state_dir", "data/state")
    ensure_dir(state_dir)

    gh = producer.gh
    
    # Load state
    repos_state_file = repo_state_path(state_dir)
    issues_state_file = issues_state_path(state_dir)
    repos_state = load_json(repos_state_file, default={})
    issues_state = load_json(issues_state_file, default={})

    # Collect repositories
    repo_sources = config.get("repo_sources", {}) or {}
    orgs = repo_sources.get("orgs", []) or []
    search_queries = repo_sources.get("search_queries", []) or []
    max_repos_total = int(repo_sources.get("max_repos_total", 200))

    logger.info(f"Collecting repositories: orgs={orgs}, queries={search_queries}")
    repos = collect_repos(
        gh,
        orgs=orgs,
        search_queries=search_queries,
        max_repos_total=max_repos_total,
    )
    stats['repos_collected'] = len(repos)
    logger.info(f"Collected {len(repos)} repositories")

    # Update repos state
    repos_state["last_run_at"] = utc_now_iso()
    repos_state["last_repos_count"] = len(repos)
    save_json(repos_state_file, repos_state)

    # Collect issues per repo
    icfg = config.get("issues_collection", {}) or {}
    default_since = str(icfg.get("default_since", "2024-01-01T00:00:00Z"))
    max_repos_per_run = int(icfg.get("max_repos_per_run", 50))
    max_pages_per_repo = int(icfg.get("max_pages_per_repo", 10))
    include_state = str(icfg.get("include_state", "all"))
    overlap_seconds = int(icfg.get("since_overlap_seconds", 600))
    collect_first_comment = bool(icfg.get("collect_first_comment", False))
    max_first_comment_fetch = int(icfg.get("max_first_comment_fetch", 0))

    repos_to_process = repos[:max_repos_per_run]
    
    for idx, repo_obj in enumerate(repos_to_process, start=1):
        repo_full = repo_obj.get("full_name")
        if not repo_full:
            continue

        # Skip repos with issues disabled
        if repo_obj.get("has_issues") is False:
            logger.info(f"[{idx}/{len(repos_to_process)}] Skipping {repo_full} (has_issues=False)")
            continue

        since = get_since(issues_state, repo_full, default_since)
        logger.info(f"[{idx}/{len(repos_to_process)}] Processing {repo_full} since={since}")

        try:
            # Collect issues using existing logic
            issues, truncated = collect_issues_for_repo(
                gh,
                repo_full_name=repo_full,
                since_iso=since,
                include_state=include_state,
                max_pages=max_pages_per_repo,
                collect_first_comment=collect_first_comment,
                max_first_comment_fetch=max_first_comment_fetch,
            )

            stats['issues_collected'] += len(issues)

            # Publish to Kafka
            if issues:
                published = producer.publish_batch(issues)
                stats['issues_published'] += published
                logger.info(f"  Published {published}/{len(issues)} issues to Kafka topic '{producer.topic}'")

                # Update cursor
                mx = max_updated_at(issues)
                if mx:
                    new_since = subtract_overlap(mx, overlap_seconds)
                    set_since(issues_state, repo_full, new_since)
            else:
                logger.info(f"  No new issues found")

            # Update repo metadata
            set_repo_run_meta(
                issues_state,
                repo_full,
                last_run_at=utc_now_iso(),
                collected_count=len(issues),
                truncated=bool(truncated),
                status="ok",
            )

            if truncated:
                logger.warning(f"  Repo truncated (max_pages reached): {repo_full}")

        except Exception as e:
            stats['errors'] += 1
            logger.error(f"  Error processing {repo_full}: {e}", exc_info=True)
            
            set_repo_run_meta(
                issues_state,
                repo_full,
                last_run_at=utc_now_iso(),
                collected_count=0,
                truncated=False,
                status=f"error: {type(e).__name__}",
            )

        # Save state after each repo (crash-safe)
        save_json(issues_state_file, issues_state)

    return stats


def main():
    """Main entry point for Kafka producer."""
    # Configuration from environment
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    topic = os.getenv('KAFKA_TOPIC', 'github.issues.raw')
    config_path = os.getenv('COLLECTOR_CONFIG', 'config.json')
    poll_interval = int(os.getenv('POLL_INTERVAL', '1800'))  # 30 minutes default

    logger.info("=" * 70)
    logger.info("GitHub Issues Kafka Producer")
    logger.info("=" * 70)
    logger.info(f"Kafka: {bootstrap_servers}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Poll Interval: {poll_interval}s")
    logger.info("=" * 70)

    # Load configuration
    config = load_config(config_path)

    # Initialize GitHub client
    logger.info("Initializing GitHub client...")
    gh = GitHubClient()

    # Initialize Kafka producer
    logger.info(f"Connecting to Kafka at {bootstrap_servers}...")
    producer = GitHubKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        github_client=gh
    )

    try:
        cycle = 0
        while True:
            cycle += 1
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Starting collection cycle {cycle}")
            logger.info("=" * 70)
            
            stats = run_collection_cycle(producer, config)
            
            logger.info("=" * 70)
            logger.info(f"Cycle {cycle} complete:")
            logger.info(f"  Repos collected: {stats['repos_collected']}")
            logger.info(f"  Issues collected: {stats['issues_collected']}")
            logger.info(f"  Issues published: {stats['issues_published']}")
            logger.info(f"  Errors: {stats['errors']}")
            logger.info("=" * 70)
            
            logger.info(f"Sleeping for {poll_interval} seconds...")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Shutting down producer (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        producer.close()


if __name__ == '__main__':
    main()