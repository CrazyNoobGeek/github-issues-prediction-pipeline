"""
Kafka producer for GitHub issues.
MODE: BACKFILL + LIVE MONITORING
"""
from __future__ import annotations

import json
import os
import time
import logging
import requests
from typing import Any, Dict, List
from datetime import datetime

from kafka import KafkaProducer

# Configuration des Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubKafkaProducer:
    """Producteur Kafka optimisé."""
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        logger.info(f"🔌 Connexion à Kafka sur {bootstrap_servers}...")
        self.producer = None
        for i in range(10):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                    key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                    acks='all',
                    retries=3
                )
                logger.info("✅ Connecté à Kafka !")
                break
            except Exception as e:
                logger.warning(f"⏳ Tentative {i+1}/10 échouée: {e}")
                time.sleep(5)
        if not self.producer:
            raise Exception("❌ Impossible de se connecter à Kafka.")

    def publish_issue(self, issue: Dict[str, Any]):
        try:
            key = str(issue.get('id', ''))
            self.producer.send(self.topic, key=key, value=issue)
            return True
        except Exception as e:
            logger.error(f"❌ Erreur envoi Kafka: {e}")
            return False
    
    def flush(self):
        if self.producer: self.producer.flush()

    def close(self):
        if self.producer: self.producer.close()

def format_issue_for_spark(issue: Dict, repo_name: str) -> Dict:
    """Nettoie et formate l'issue pour Spark."""
    issue['repo_full_name'] = repo_name
    if 'user' in issue and isinstance(issue['user'], dict):
        issue['user_login'] = issue['user'].get('login')
    else: issue['user_login'] = None
    
    if 'labels' in issue and isinstance(issue['labels'], list):
        issue['labels'] = [l.get('name') for l in issue['labels'] if isinstance(l, dict)]
    else: issue['labels'] = []

    if 'assignees' in issue and isinstance(issue['assignees'], list):
        issue['assignees'] = [a.get('login') for a in issue['assignees'] if isinstance(a, dict)]
    else: issue['assignees'] = []
    return issue

def fetch_batch_issues(repo: str, token: str, page: int, since_date: str) -> List[Dict]:
    """Récupère une page spécifique d'issues."""
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {
        "state": "all",                 # Open + Closed
        "since": since_date,            # Date de début
        "sort": "created",
        "direction": "asc",             # IMPORTANT : On récupère du plus vieux au plus récent pour le backfill
        "per_page": 100,                # Max par page
        "page": page                    # Numéro de page
    }
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token: headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403:
            logger.warning("⚠️ Rate Limit GitHub atteint. Pause de 60s...")
            time.sleep(60)
            return []
        else:
            logger.error(f"❌ Erreur API {resp.status_code}: {resp.text}")
            return []
    except Exception as e:
        logger.error(f"❌ Erreur Réseau: {e}")
        return []

def main():
    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    topic = os.getenv('KAFKA_TOPIC', 'github.issues.raw')
    repo_to_monitor = os.getenv('TARGET_REPO', 'kubernetes/kubernetes')
    github_token = os.getenv('GITHUB_TOKEN', '')
    poll_interval = int(os.getenv('POLL_INTERVAL', '60'))
    
    # DATE DE DÉBUT
    START_DATE = "2026-01-01T00:00:00Z"

    producer = GitHubKafkaProducer(kafka_servers, topic)
    seen_ids = set()

    logger.info(f"🚀 DÉMARRAGE : Récupération de TOUT l'historique depuis {START_DATE}")

    # --- PHASE 1 : BACKFILL (HISTORIQUE) ---
    page = 1
    total_backfill = 0
    
    while True:
        logger.info(f"📚 Backfill - Récupération page {page}...")
        issues = fetch_batch_issues(repo_to_monitor, github_token, page, START_DATE)
        
        if not issues:
            logger.info("✅ Fin du Backfill (plus d'issues trouvées).")
            break
        
        count = 0
        for raw_issue in issues:
            i_id = raw_issue.get('id')
            if i_id not in seen_ids:
                formatted = format_issue_for_spark(raw_issue, repo_to_monitor)
                producer.publish_issue(formatted)
                seen_ids.add(i_id)
                count += 1
        
        producer.flush()
        total_backfill += count
        logger.info(f"📥 Page {page} : {count} issues envoyées (Total: {total_backfill})")
        
        if len(issues) < 100:
            logger.info("✅ Fin du Backfill (Dernière page incomplète).")
            break
            
        page += 1
        time.sleep(1) # Petite pause pour être gentil avec l'API XD

    logger.info("=" * 60)
    logger.info(f"🎉 HISTORIQUE TERMINÉ. {total_backfill} issues chargées.")
    logger.info("👀 PASSAGE EN MODE LIVE MONITORING (Nouveaux tickets uniquement)")
    logger.info("=" * 60)

    # --- PHASE 2 : LIVE MONITORING (BOUCLE INFINIE) ---
    try:
        while True:
            # En mode live, on demande juste la page 1 triée par 'desc' (les plus récents)
            # pour attraper ce qui vient d'arriver pendant qu'on dormait
            logger.info(f"🔍 Vérification des nouveautés...")
            
            # Appel manuel simplifié pour le live (juste les derniers)
            url = f"https://api.github.com/repos/{repo_to_monitor}/issues"
            params = {"state": "all", "sort": "created", "direction": "desc", "per_page": 20}
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token: headers["Authorization"] = f"token {github_token}"
            
            try:
                resp = requests.get(url, headers=headers, params=params)
                live_issues = resp.json() if resp.status_code == 200 else []
            except: live_issues = []

            new_count = 0
            for raw_issue in live_issues:
                i_id = raw_issue.get('id')
                if i_id not in seen_ids:
                    formatted = format_issue_for_spark(raw_issue, repo_to_monitor)
                    producer.publish_issue(formatted)
                    seen_ids.add(i_id)
                    new_count += 1
                    logger.info(f"🔔 LIVE : #{raw_issue.get('number')} détecté !")
            
            if new_count > 0:
                producer.flush()
                logger.info(f"📨 {new_count} nouveaux tickets envoyés.")
            else:
                logger.info("💤 Rien à signaler.")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("🛑 Arrêt.")
    finally:
        producer.close()

if __name__ == '__main__':
    main()