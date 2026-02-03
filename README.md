# GitHub Issues Prediction Pipeline

A scalable Big Data pipeline for collecting, streaming, processing, and analyzing GitHub issues using Kafka, Spark, and MongoDB.

## Architecture

```
GitHub API → Collector → Kafka (github.issues.raw) → Spark Streaming → MongoDB (github.issues)
```

### Components

- **Collector** (`github_bigdata_pipeline/collector/`): Robust GitHub API client with incremental state management
- **Kafka Producer** (`produce_issues.py`): Streams issues to Kafka using existing collector logic
- **Kafka**: Message broker for fault-tolerant streaming (Confluent 7.5.0)
- **Spark Streaming** (`spark/process_stream.py`): Real-time data processing with feature engineering (Bitnami 3.5)
- **MongoDB**: NoSQL storage for processed issues with indexes (MongoDB 7)

## Prerequisites

- Docker & Docker Compose
- GitHub Personal Access Token ([create one here](https://github.com/settings/tokens))
- At least 8GB RAM available for Docker

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd github-issues-prediction-pipeline

# Copy environment template
cp .env.example .env

# Edit .env and add your GitHub token
nano .env  # or use your preferred editor
```

**Required in `.env`:**
```bash
GITHUB_TOKEN=ghp_your_actual_github_token_here
```

### 2. Configure Data Collection

Edit `config.json` to specify which repositories to collect:

```json
{
  "state_dir": "data/state",
  "repo_sources": {
    "orgs": ["tensorflow", "pytorch"],
    "search_queries": [
      "stars:>10000 language:python",
      "stars:>5000 topic:machine-learning"
    ],
    "max_repos_total": 100
  },
  "issues_collection": {
    "default_since": "2024-01-01T00:00:00Z",
    "max_repos_per_run": 20,
    "max_pages_per_repo": 10,
    "include_state": "all",
    "since_overlap_seconds": 600,
    "collect_first_comment": false
  }
}
```

### 3. Start Infrastructure

```bash
# Start all services (Kafka, MongoDB, Spark, Producer)
docker-compose up -d

# Check service health
docker-compose ps

# View producer logs
docker-compose logs -f kafka-producer
```

### 4. Start Spark Streaming Job

```bash
# Make submit script executable
chmod +x spark/submit.sh

# Submit Spark streaming job
./spark/submit.sh
```

## Monitoring

### Service UIs

- **Spark Master UI**: http://localhost:8080
- **Spark Job UI**: http://localhost:4040
- **Kafka**: localhost:9092 (external) / kafka:29092 (internal)
- **MongoDB**: localhost:27017

### Viewing Logs

```bash
# Producer logs
docker-compose logs -f kafka-producer

# Spark master logs
docker-compose logs -f spark-master

# Spark worker logs
docker-compose logs -f spark-worker

# All services
docker-compose logs -f
```

### MongoDB Queries

```bash
# Connect to MongoDB
docker exec -it mongodb mongosh -u root -p rootpassword

# In mongosh:
use github

# Count total issues
db.issues.count()

# View a sample issue
db.issues.findOne()

# Count by state
db.issues.aggregate([
  { $group: { _id: "$state", count: { $sum: 1 } } }
])

# Top repositories by issue count
db.issues.aggregate([
  { $group: { _id: "$repo_full_name", count: { $sum: 1 } } },
  { $sort: { count: -1 } },
  { $limit: 10 }
])

# Issues with bugs
db.issues.find({ is_bug: true }).count()
```

### Kafka Topics

```bash
# List topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:29092 --list

# View topic details
docker exec -it kafka kafka-topics --bootstrap-server localhost:29092 --describe --topic github.issues.raw

# Consume messages (first 5)
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:29092 \
  --topic github.issues.raw \
  --from-beginning \
  --max-messages 5
```

## Data Flow Details

### 1. Producer (`kafka-producer` container)

**What it does:**
- Runs `github_bigdata_pipeline/collector/produce_issues.py`
- Uses your existing `GitHubClient` and collector modules
- Discovers repositories from orgs and search queries
- Fetches issues incrementally with state tracking
- Publishes issues to Kafka topic `github.issues.raw`
- Runs every 30 minutes by default (configurable via `POLL_INTERVAL`)

**State Management:**
- Maintains state in `data/state/issues_state.json`
- Tracks `since` timestamp per repository for incremental collection
- Crash-safe: saves state after each repository

### 2. Kafka

**What it does:**
- Buffers issues in `github.issues.raw` topic
- Provides fault-tolerant message queue
- Enables decoupling of producer and consumer
- Supports replay and multiple consumers

### 3. Spark Streaming (`spark/process_stream.py`)

**What it does:**
- Reads from Kafka topic in micro-batches (every 30 seconds)
- Applies schema validation matching your `keep_issue()` output
- Enriches data with calculated features:
  - `label_count`, `has_labels`
  - `body_length`, `title_length`
  - `time_to_close_hours`
  - `time_to_first_comment_hours`
  - `age_hours`
  - Boolean flags: `is_bug`, `is_enhancement`, `is_documentation`, etc.
- Converts timestamp strings to proper datetime types
- Writes batches to MongoDB

### 4. MongoDB

**What it does:**
- Stores processed issues in `github.issues` collection
- Indexed for efficient querying
- Supports aggregation and analytics

## Project Structure

```
github-issues-prediction-pipeline/
├── github_bigdata_pipeline/        # Your existing collector package
│   └── collector/
│       ├── github_client.py        # Robust GitHub API client
│       ├── repos.py                # Repository discovery
│       ├── issues.py               # Issue collection
│       ├── state.py                # State management
│       ├── io_utils.py             # File utilities
│       └── produce_issues.py       # NEW: Kafka producer
│
├── spark/                          # Spark streaming jobs
│   ├── process_stream.py           # Main streaming job
│   ├── requirements.txt
│   └── submit.sh                   # Submit helper script
│
├── mongo/                          # MongoDB initialization
│   └── init-scripts/
│       └── 01-init-db.js           # Index creation
│
├── data/                           # Local data (gitignored)
│   └── state/                      # Collection state files
│
├── docker-compose.yml              # Infrastructure orchestration
├── Dockerfile.producer             # Producer container
├── requirements-producer.txt       # Producer dependencies
├── config.json                     # Collector configuration
├── .env                            # Environment variables
├── .env.example                    # Environment template
└── README.md                       # This file
```

## Development

### Running Original Collector Standalone

Your original collector still works independently:

```bash
# Set environment
export GITHUB_TOKEN=your_token_here
export COLLECTOR_CONFIG=config.json

# Run standalone (writes to data/raw/)
python run_collector.py
```

### Modifying Producer Logic

```bash
# Edit producer
vim github_bigdata_pipeline/collector/produce_issues.py

# Rebuild and restart
docker-compose up -d --build kafka-producer

# View logs
docker-compose logs -f kafka-producer
```

### Modifying Spark Job

```bash
# Edit streaming job
vim spark/process_stream.py

# Resubmit to Spark (will restart the job)
./spark/submit.sh
```

### Modifying Collector Core Logic

If you update the core collector modules (`github_client.py`, `issues.py`, etc.):

```bash
# Rebuild producer (it imports these modules)
docker-compose up -d --build kafka-producer

# No need to rebuild Spark (it only reads from Kafka)
```

### Testing Locally

```bash
# Test producer imports
docker exec -it kafka-producer python -c "from github_bigdata_pipeline.collector import GitHubClient; print('OK')"

# Test Spark can access code
docker exec -it spark-master ls -la /app/spark/

# Test MongoDB connection
docker exec -it mongodb mongosh -u root -p rootpassword --eval "db.version()"
```

## Configuration

### Producer Settings (`.env`)

```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:29092   # Kafka connection
KAFKA_TOPIC=github.issues.raw         # Topic name
GITHUB_TOKEN=ghp_xxx                  # Your GitHub token
POLL_INTERVAL=1800                    # Collection cycle (seconds)
COLLECTOR_CONFIG=config.json          # Config file path
```

### Collection Settings (`config.json`)

```json
{
  "repo_sources": {
    "orgs": ["tensorflow"],           // GitHub orgs to scan
    "search_queries": [               // Search queries
      "stars:>10000 language:python"
    ],
    "max_repos_total": 100            // Max repos to collect
  },
  "issues_collection": {
    "default_since": "2024-01-01T00:00:00Z",  // Start date
    "max_repos_per_run": 20,          // Repos per cycle
    "max_pages_per_repo": 10,         // Pages per repo
    "include_state": "all",           // "all", "open", or "closed"
    "since_overlap_seconds": 600,     // Overlap for cursor
    "collect_first_comment": false    // Fetch first comment timestamp
  }
}
```

### Spark Resources

Edit `docker-compose.yml`:

```yaml
spark-worker:
  environment:
    - SPARK_WORKER_MEMORY=8G   # Increase memory
    - SPARK_WORKER_CORES=4     # Increase cores
```

## Troubleshooting

### Producer Not Starting

```bash
# Check logs
docker-compose logs kafka-producer

# Verify GitHub token
docker exec -it kafka-producer env | grep GITHUB_TOKEN

# Test GitHub connection
docker exec -it kafka-producer python -c "
from github_bigdata_pipeline.collector import GitHubClient
gh = GitHubClient()
print('GitHub client OK')
"
```

### Kafka Connection Issues

```bash
# Check Kafka is running
docker-compose logs kafka

# Verify topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:29092 --list

# Test producer connection
docker exec -it kafka-producer python -c "
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers='kafka:29092')
print('Kafka OK')
producer.close()
"
```

### Spark Job Failures

```bash
# Check Spark master
docker-compose logs spark-master

# Check Spark worker
docker-compose logs spark-worker

# Verify Spark can reach Kafka
docker exec -it spark-master ping kafka

# Verify Spark can reach MongoDB
docker exec -it spark-master ping mongodb
```

### No Data in MongoDB

```bash
# Check producer is publishing
docker-compose logs kafka-producer | grep "Published"

# Check Kafka has messages
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:29092 \
  --topic github.issues.raw \
  --from-beginning \
  --max-messages 1

# Check Spark is processing
docker-compose logs spark-master | grep "Batch"

# Check MongoDB
docker exec -it mongodb mongosh -u root -p rootpassword --eval "use github; db.issues.count()"
```

## Stopping Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (DELETES ALL DATA)
docker-compose down -v

# Stop specific service
docker-compose stop kafka-producer
```

## Data Backup

```bash
# Export MongoDB data
docker exec mongodb mongodump \
  --uri="mongodb://root:rootpassword@localhost:27017" \
  --db=github \
  --out=/data/backup

# Copy backup
docker cp mongodb:/data/backup ./mongodb-backup
```

## Performance Tuning

### For High-Volume Collection

1. **Increase Kafka retention:**
   ```yaml
   # docker-compose.yml
   KAFKA_LOG_RETENTION_HOURS: 720  # 30 days
   ```

2. **Increase Spark batch size:**
   ```python
   # spark/process_stream.py
   .option("maxOffsetsPerTrigger", "5000")  # Process more per batch
   ```

3. **Reduce polling interval:**
   ```bash
   # .env
   POLL_INTERVAL=600  # 10 minutes
   ```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.