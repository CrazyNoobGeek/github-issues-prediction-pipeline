
"""
Spark Structured Streaming job for processing GitHub issues from Kafka.
Reads from 'github.issues.raw' topic and writes cleaned data to MongoDB.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, to_timestamp, size, when, length,
    expr, current_timestamp, unix_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    BooleanType, ArrayType, LongType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "GitHubIssuesStreamProcessor") -> SparkSession:
    """Create and configure Spark session with Kafka and MongoDB connectors."""
    logger.info("Creating Spark session...")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0") \
        .config("spark.mongodb.write.connection.uri",
                "mongodb://root:rootpassword@mongodb:27017") \
        .config("spark.mongodb.write.database", "github") \
        .config("spark.mongodb.write.collection", "issues") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully")
    return spark


def define_issue_schema() -> StructType:
    """
    Define schema matching the output from github_bigdata_pipeline.collector.issues.keep_issue().
    This must match exactly what your collector produces.
    """
    return StructType([
        StructField("collected_at", StringType(), True),
        StructField("repo_full_name", StringType(), True),
        
        # Keys
        StructField("id", LongType(), True),
        StructField("node_id", StringType(), True),
        StructField("number", IntegerType(), True),
        
        # Resolution signals
        StructField("state", StringType(), True),
        StructField("state_reason", StringType(), True),
        
        # Text features
        StructField("title", StringType(), True),
        StructField("body", StringType(), True),
        
        # User metadata
        StructField("user_login", StringType(), True),
        StructField("author_association", StringType(), True),
        
        # Labels & activity
        StructField("labels", ArrayType(StringType()), True),
        StructField("comments", IntegerType(), True),
        
        # Timestamps
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("closed_at", StringType(), True),
        
        # Assignment
        StructField("assignees", ArrayType(StringType()), True),
        StructField("num_assignees", IntegerType(), True),
        StructField("locked", BooleanType(), True),
        StructField("active_lock_reason", StringType(), True),
        StructField("milestone", StringType(), True),
        
        # URLs
        StructField("html_url", StringType(), True),
        StructField("url", StringType(), True),
        
        # Flags
        StructField("is_pull_request", BooleanType(), True),
        StructField("first_comment_at", StringType(), True),
    ])


def clean_and_enrich(df):
    """
    Apply data cleaning, validation, and feature engineering.
    """
    logger.info("Applying data transformations...")
    
    return df \
        .filter(col("id").isNotNull()) \
        .filter(col("repo_full_name").isNotNull()) \
        .filter(col("number").isNotNull()) \
        .withColumn("collected_at", to_timestamp(col("collected_at"))) \
        .withColumn("created_at", to_timestamp(col("created_at"))) \
        .withColumn("updated_at", to_timestamp(col("updated_at"))) \
        .withColumn("closed_at", to_timestamp(col("closed_at"))) \
        .withColumn("first_comment_at", to_timestamp(col("first_comment_at"))) \
        .withColumn("processed_at", current_timestamp()) \
        .withColumn("label_count", size(col("labels"))) \
        .withColumn("has_labels", col("label_count") > 0) \
        .withColumn("has_assignees", col("num_assignees") > 0) \
        .withColumn("has_body", 
                   when(col("body").isNotNull(), length(col("body")) > 0).otherwise(False)) \
        .withColumn("body_length", 
                   when(col("body").isNotNull(), length(col("body"))).otherwise(0)) \
        .withColumn("title_length", 
                   when(col("title").isNotNull(), length(col("title"))).otherwise(0)) \
        .withColumn("is_closed", col("state") == "closed") \
        .withColumn("is_bug", 
                   expr("array_contains(labels, 'bug')")) \
        .withColumn("is_enhancement", 
                   expr("array_contains(labels, 'enhancement')")) \
        .withColumn("is_documentation", 
                   expr("array_contains(labels, 'documentation')")) \
        .withColumn("is_question", 
                   expr("array_contains(labels, 'question')")) \
        .withColumn(
            "time_to_close_hours",
            when(
                col("closed_at").isNotNull(),
                (unix_timestamp(col("closed_at")) - unix_timestamp(col("created_at"))) / 3600
            ).otherwise(None)
        ) \
        .withColumn(
            "time_to_first_comment_hours",
            when(
                col("first_comment_at").isNotNull(),
                (unix_timestamp(col("first_comment_at")) - unix_timestamp(col("created_at"))) / 3600
            ).otherwise(None)
        ) \
        .withColumn(
            "age_hours",
            (unix_timestamp(current_timestamp()) - unix_timestamp(col("created_at"))) / 3600
        )


def write_to_mongodb_batch(batch_df, batch_id):
    """
    Write a batch to MongoDB with error handling and logging.
    Uses upsert to avoid duplicates based on issue ID.
    """
    try:
        count = batch_df.count()
        if count == 0:
            logger.info(f"Batch {batch_id}: No records to write")
            return
        
        logger.info(f"Batch {batch_id}: Writing {count} records to MongoDB")
        
        # Write to MongoDB with upsert based on 'id' field
        batch_df.write \
            .format("mongodb") \
            .mode("append") \
            .option("database", "github") \
            .option("collection", "issues") \
            .option("replaceDocument", "false") \
            .save()
        
        logger.info(f"Batch {batch_id}: Successfully written {count} records")
        
        # Log sample record for debugging
        sample = batch_df.select("repo_full_name", "number", "state", "label_count").first()
        if sample:
            logger.info(f"Batch {batch_id}: Sample - {sample.repo_full_name}#{sample.number} "
                       f"state={sample.state} labels={sample.label_count}")
        
    except Exception as e:
        logger.error(f"Batch {batch_id}: Error writing to MongoDB: {e}", exc_info=True)
        raise


def main():
    """Main entry point for Spark streaming job."""
    logger.info("=" * 70)
    logger.info("GitHub Issues Stream Processor")
    logger.info("=" * 70)
    
    # Create Spark session
    spark = create_spark_session()
    
    # Define schema
    schema = define_issue_schema()
    
    # Kafka configuration
    kafka_bootstrap_servers = "kafka:29092"
    kafka_topic = "github.issues.raw"
    
    logger.info(f"Kafka Bootstrap Servers: {kafka_bootstrap_servers}")
    logger.info(f"Kafka Topic: {kafka_topic}")
    logger.info(f"MongoDB: mongodb://mongodb:27017/github.issues")
    logger.info("=" * 70)
    
    # Read from Kafka
    logger.info("Connecting to Kafka stream...")
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .option("maxOffsetsPerTrigger", "1000") \
        .load()
    
    logger.info("Connected to Kafka stream")
    
    # Parse JSON from Kafka value
    logger.info("Parsing JSON messages...")
    parsed_df = kafka_df.select(
        col("key").cast("string").alias("kafka_key"),
        col("timestamp").alias("kafka_timestamp"),
        col("partition").alias("kafka_partition"),
        col("offset").alias("kafka_offset"),
        from_json(col("value").cast("string"), schema).alias("data")
    ).select(
        "kafka_key", 
        "kafka_timestamp", 
        "kafka_partition", 
        "kafka_offset", 
        "data.*"
    )
    
    # Clean and enrich data
    cleaned_df = clean_and_enrich(parsed_df)
    
    # Write to MongoDB using foreachBatch for better control
    logger.info("Starting streaming query...")
    query = cleaned_df.writeStream \
        .foreachBatch(write_to_mongodb_batch) \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/spark-checkpoint-github-issues") \
        .trigger(processingTime="30 seconds") \
        .start()
    
    logger.info("=" * 70)
    logger.info("Streaming query started successfully!")
    logger.info(f"Spark UI: http://localhost:4040")
    logger.info(f"Spark Master UI: http://localhost:8080")
    logger.info("Processing GitHub issues from Kafka to MongoDB...")
    logger.info("=" * 70)
    
    # Wait for termination
    query.awaitTermination()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise