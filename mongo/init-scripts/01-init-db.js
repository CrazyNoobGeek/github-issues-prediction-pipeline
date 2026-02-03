
// MongoDB initialization script for GitHub issues database

db = db.getSiblingDB('github');

print('Initializing GitHub issues database...');

// Create issues collection
db.createCollection('issues');

// Create indexes for efficient querying
print('Creating indexes...');

// Primary key index (unique)
db.issues.createIndex({ "id": 1 }, { unique: true, name: "idx_id" });

// Query indexes
db.issues.createIndex({ "repo_full_name": 1 }, { name: "idx_repo" });
db.issues.createIndex({ "number": 1 }, { name: "idx_number" });
db.issues.createIndex({ "state": 1 }, { name: "idx_state" });
db.issues.createIndex({ "created_at": 1 }, { name: "idx_created_at" });
db.issues.createIndex({ "updated_at": 1 }, { name: "idx_updated_at" });
db.issues.createIndex({ "is_pull_request": 1 }, { name: "idx_is_pr" });

// Compound indexes for common queries
db.issues.createIndex({ "repo_full_name": 1, "number": 1 }, { name: "idx_repo_number" });
db.issues.createIndex({ "repo_full_name": 1, "state": 1 }, { name: "idx_repo_state" });
db.issues.createIndex({ "repo_full_name": 1, "created_at": -1 }, { name: "idx_repo_created" });
db.issues.createIndex({ "state": 1, "created_at": -1 }, { name: "idx_state_created" });

// Label and feature indexes
db.issues.createIndex({ "labels": 1 }, { name: "idx_labels" });
db.issues.createIndex({ "is_bug": 1 }, { name: "idx_is_bug" });
db.issues.createIndex({ "is_enhancement": 1 }, { name: "idx_is_enhancement" });

print('GitHub database initialized successfully!');
print('Indexes created:');
db.issues.getIndexes().forEach(function(idx) {
    print('  - ' + idx.name);
});