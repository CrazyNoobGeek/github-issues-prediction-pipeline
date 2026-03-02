import pymongo
from datetime import datetime

# Connect to MongoDB (using the internal Docker network credentials)
client = pymongo.MongoClient("mongodb://root:rootpassword@mongodb:27017/")
db = client["github"]
collection = db["issues"]

print("\n" + "="*60)
print(f"📊  GITHUB DATA ANALYTICS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

# 1.Total Count
total = collection.count_documents({})
print(f"Total Issues Stored: {total:,}")
print("-" * 60)

# 2. Aggregation: Top 5 Repositories by Issue Count
pipeline = [
    {"$group": {"_id": "$repo_full_name", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 5}
]

print("🏆  TOP 5 REPOSITORIES (by volume):")
print(f"{'Repository':<40} | {'Count':<10}")
print("-" * 55)
for doc in collection.aggregate(pipeline):
    print(f"{doc['_id']:<40} | {doc['count']:<10}")

print("-" * 60)

# 3. Simple Keyword Classification (Bugs vs Features)
bug_count = collection.count_documents({"labels": {"$regex": "bug", "$options": "i"}})
feat_count = collection.count_documents({"labels": {"$regex": "enhancement|feature", "$options": "i"}})

print("🏷️   CLASSIFICATION:")
print(f"🐛  Identified Bugs:        {bug_count:,}")
print(f"✨  Feature Requests:       {feat_count:,}")
print("="*60 + "\n")
