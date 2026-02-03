
#!/bin/bash

echo "=" | tr '=' '=' | head -c 70; echo
echo "Submitting Spark Streaming Job"
echo "=" | tr '=' '=' | head -c 70; echo

docker exec -it spark-master spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:10.4.0 \
  --conf spark.mongodb.write.connection.uri=mongodb://root:rootpassword@mongodb:27017 \
  --conf spark.mongodb.write.database=github \
  --conf spark.mongodb.write.collection=issues \
  --conf spark.executor.memory=2g \
  --conf spark.driver.memory=1g \
  --conf spark.sql.streaming.checkpointLocation=/tmp/spark-checkpoint-github-issues \
  /app/spark/process_stream.py

echo "=" | tr '=' '=' | head -c 70; echo
echo "Spark job submitted!"
echo "Monitor at: http://localhost:8080 (Master) and http://localhost:4040 (Job)"
echo "=" | tr '=' '=' | head -c 70; echo