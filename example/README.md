# example

TODO

## How to run

1. **Build fat jar**
   - `$ sbt assembly`
2. **Run as spark-job**
   - submit to cluster or run locally

**Example local execution**
```
$SPARK_HOME/bin/spark-submit --class "limmen.fasttext_on_spark.Main" target/scala-2.11/fasttext_on_spark-assembly-0.1.0-SNAPSHOT.jar --input "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/wiki/clean.txt" --output "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/output" --cluster --partitions 2 --iterations 300 --dim 300 --windowsize 5 --minn 3 --maxn 6 --mincount 1 --norm --bucket 10000
```

**Example cluster execution**

``` shell
$SPARK_HOME/bin/spark-submit \
    --master spark://limmen-MS-7823:7077 \
    --class "limmen.fasttext_on_spark.Main" \
    --conf spark.cores.max=8 \
    --conf spark.task.cpus=1 \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    --executor-memory 2g \
    --driver-memory 2g \
    /media/limmen/HDD/workspace/scala/fasttext_on_spark/target/scala-2.11/fasttext_on_spark-assembly-0.1.0-SNAPSHOT.jar --input "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/clean2_corpus.txt" --output "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/output" --cluster --partitions 2 --iterations 300 --dim 300 --windowsize 5 --minn 3 --maxn 6 --mincount 1 --norm --bucket 10000 --verbose --average
```

## Parameters

| Parameter-name   | Description                                                                   |
| -----            | -----------                                                                   |
| --input          | [String] Path to input corpus                                                 |
| --output         | [String] Path to save output vectors                                          |
| --dim            | [Int] Dimension of vectors [100]                                              |
| --lr             | [Double] Learning rate for training [0.025]                                   |
| --partitions     | [Int] Number of Spark partitions [1]                                          |
| --iterations     | [Int] Number of iterations for training [1]                                   |
| --mincount       | [Int] Min frequency of words for training [5]                                 |
| --sentencelength | [Int] Maximum sentence length (otherwise split into multiple sentences [1000] |
| --windowsize     | [Int] Size of window of context words [5]                                     |
| --saveparallel   | [Boolean] If true, output is saved in partitioned format (faster) [false]     |
| --cluster        | [Boolean] If true spark is ran in cluster mode [false]                        |
| --minn           | [Int] Min length of n-grams 3                                                 |
| --maxn           | [Int] Max length of n-grams 3                                                 |
| --bucket         | [Int] Number of buckets for hashing n-grams [2000000]                         |
| --norm           | [Boolean] If true output vectors are first normalized with L2 norm [false]    |
| --average        | [Boolean] If true paritioned embeddings are averaged instead of summed        |
| --verbose        | [Boolean] If true logging is verbose [false]                                  |

## License

BSD 2-clause, see [LICENSE](./LICENSE)

## Author

Kim Hammar, [kimham@kth.se](mailto:kimham@kth.se)
