#!/bin/bash

#$SPARK_HOME/bin/spark-submit \
#    --master spark://limmen-MS-7823:7077 \
#    --class "limmen.fasttext_on_spark.Main" \
#    --conf spark.cores.max=8 \
#    --conf spark.task.cpus=7 \
#    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
#    --conf spark.rpc.message.maxSize=2000 \
#    --executor-memory 8g \
#    --driver-memory 8g \
#/home/kim/workspace/scala/FastTextOnSpark/target/scala-2.11/fasttext_on_spark-assembly-1.0.0.jar --input "/home/kim/workspace/scala/fasttext_on_spark/data/clean2_corpus.txt" --output "/home/kim/workspace/scala/fasttext_on_spark/data/output" --cluster --partitions 5 --iterations 5 --saveparallel --dim 100 --windowsize 5 --algorithm "fasttext" --minn 3 --maxn 6 --norm

#/home/kim/workspace/scala/FastTextOnSpark/target/scala-2.11/fasttext_on_spark-assembly-1.0.0.jar --input "/home/kim/workspace/scala/FastTextOnSpark/data/input" --output "/home/kim/workspace/scala/FastTextOnSpark/data/output" --cluster --partitions 20 --iterations 2 --saveparallel --dim 100 --windowsize 5 --algorithm "fasttext" --minn 3 --maxn 6 --norm

$SPARK_HOME/bin/spark-submit --class "com.github.limmen.spark.example.Main" target/scala-2.11/example-assembly-1.0.0.jar --input "/home/kim/workspace/scala/FastTextOnSpark/data/input" --output "/home/kim/workspace/scala/FastTextOnSpark/data/output" --cluster --partitions 2 --iterations 30 --dim 300 --windowsize 5 --minn 3 --maxn 6 --mincount 1 --norm --bucket 10000
