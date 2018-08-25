package com.github.limmen.spark.example

import com.github.limmen.spark.mllib.fasttext.FastText
import language.postfixOps
import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.rogach.scallop.ScallopConf

/**
 * Parser of command-line arguments
 */
class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val input = opt[String](required = true, descr = "input corpus path")
  val output = opt[String](required = true, descr = "output folder path for saved word embeddings")
  val dim = opt[Int](required = true, validate = (0<), descr = "dimension of word embeddings [100]", default = Some(100))
  val lr = opt[Double](required = true, validate = (0<), default = Some(0.025), descr = "learning rate [0.025]")
  val partitions = opt[Int](required = true, validate = (0<), default = Some(1), descr = "number of partitions to distribute training on [1]")
  val iterations = opt[Int](required = true, validate = (0<), default = Some(1), descr = "number of iterations for training [1]")
  val mincount = opt[Int](required = true, validate = (0<), default = Some(5), descr = "min word count, exclude all word from training set with lesser frequency [5]")
  val sentencelength = opt[Int](required = true, validate = (0<), default = Some(1000), descr = "Max length of a given sentence (longer sentences are split) [1000]")
  val windowsize = opt[Int](required = true, validate = (0<), default = Some(5), descr = "Context window size for training [5]")
  val saveparallel = opt[Boolean](descr = "Flag set to true means that output is saved in parallel format with Spark")
  val cluster = opt[Boolean](descr = "Flag set to true means that the application is running in cluster mode, otherwise it runs locally")
  val minn = opt[Int](required = true, validate = (0<), default = Some(3), descr = "Minimum length of char n-gram [3]")
  val maxn = opt[Int](required = true, validate = (0<), default = Some(6), descr = "Maximum length of char n-gram [3]")
  val bucket = opt[Int](required = true, validate = (0<), default = Some(2000000), descr = "Number of buckets for n-gram hashing [2000000]")
  val norm = opt[Boolean](descr = "Flag to set if output should be L2-normalized vectors")
  val average = opt[Boolean](descr = "Flag to set if partitions should be averaged (otherwise they are combined by sum operation)")
  val verbose = opt[Boolean](descr = "Flag to set if logging should be verbose")
  verify()
}

/**
 * Project entrypoint, orchestrates the pipeline for training word vectors.
 *
 * 1. Parse cmd arguments
 * 2. Setup spark
 * 3. Read input corpus
 * 4. Perform distributed training of word vectors (FastText or Word2Vec)
 * 5. Save output vectors in word2vec textual format
 * 6. Save training time results
 *
 * @author Kim Hammar <kimham@kth.se> <limmen@github>
 */
object Main {

  def main(args: Array[String]): Unit = {

    // Setup logging
    val log = LogManager.getRootLogger()
    log.setLevel(Level.INFO)

    log.info(s"Starting FastText Training")

    //Parse cmd arguments
    val conf = new Conf(args)

    //Save the configuration string
    val argsStr = printArgs(conf, log)

    // Setup Spark
    var sparkConf: SparkConf = null
    if (conf.cluster()) {
      sparkConf = sparkClusterSetup()
    } else {
      sparkConf = localSparkSetup()
    }

    val sc = new SparkContext(sparkConf)

    val clusterStr = sc.getConf.toDebugString
    log.info(s"Cluster settings: \n" + clusterStr)

    //Read input corpus
    val input = sc.textFile(conf.input()).map(line => line.split(" ").toSeq)

    //Run FastText Training
    val (seconds, t0) = fastText(conf, input, log, sc)
    //Save training stats
    sc.parallelize(Seq(s"Training time: ${seconds} seconds", argsStr, clusterStr)).coalesce(1).saveAsTextFile(conf.output() + "/" + t0 + "_stats")
  }

  /**
   * Perform FastText training with given spark cluster, conf and input RDD.
   */
  def fastText[S <: Iterable[String]](conf: Conf, input: RDD[S], log: Logger, sc: SparkContext): (Double, Long) = {
    //Setup FastText parameters
    val fastText = new FastText()
    val ft = fastText.setMinCount(conf.mincount()).setWindowSize(conf.windowsize()).setVectorSize(conf.dim()).setLearningRate(conf.lr()).setNumPartitions((conf.partitions())).setNumIterations(conf.iterations()).setMaxSentenceLength(conf.sentencelength()).setMinn(conf.minn()).setMaxn(conf.maxn()).setBucket(conf.bucket()).setAverage(conf.average()).setVerbose(conf.verbose())

    //Train word embeddings and measure training time
    val t0 = System.nanoTime()
    val model = ft.train(input)
    val t1 = System.nanoTime()
    val seconds = (t1 - t0) / 1000000000.0;
    //Save wordvectors
    log.info(s"Training completed in ${seconds} seconds")
    val numSimilar = 3
    val exampleWord = "anarchism"
    log.info(s"$numSimilar most similar words to the word '$exampleWord':")
    model.findSynonyms(exampleWord, numSimilar).foreach(log.info)
    log.info(s"saving word vectors to path: ${conf.output()}")
    model.saveToWord2VecFormat(model.wordVectors, conf.output() + "/" + t0 + "_vec", sc, conf.saveparallel())
    //Return training time
    (seconds, t0)
  }

  /**
   * Hard coded settings for local spark training
   *
   * @return spark configuration
   */
  def localSparkSetup(): SparkConf = {
    new SparkConf().setAppName("FastTextOnSpark").setMaster("local[*]")
  }

  /**
   * Hard coded settings for cluster spark training
   *
   * @return spark configuration
   */
  def sparkClusterSetup(): SparkConf = {
    new SparkConf().setAppName("FastTextOnSpark").set("spark.executor.heartbeatInterval", "20s").set("spark.rpc.message.maxSize", "512").set("spark.kryoserializer.buffer.max", "1024")
  }

  /**
   * Utility function for printing training configuration
   *
   * @param conf command line arguments
   * @param log logger
   * @return configuration string
   */
  def printArgs(conf: Conf, log: Logger): String = {
    val argsStr = s"Args:  | input: ${conf.input()} | output: ${conf.output()} | output: ${conf.output()} | dim: ${conf.dim()} | lr: ${conf.lr()} | partitions: ${conf.partitions()} | iterations: ${conf.iterations()} | mincount: ${conf.mincount()} | sentencelength: ${conf.sentencelength()} | windowsize: ${conf.windowsize()} | saveparallel: ${conf.saveparallel()} | cluster: ${conf.cluster()} | minn: ${conf.minn()} | maxn: ${conf.maxn()} | bucket: ${conf.bucket} | norm: ${conf.norm()} | average: ${conf.average()} | verbose: ${conf.verbose()}"
    log.info(argsStr)
    argsStr
  }
}
