package com.github.limmen.spark.mllib.fasttext

import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import org.apache.spark.sql.SparkSession
import org.scalatest.PrivateMethodTester._
import org.scalatest._

class FastTextSuite extends FunSuite with Matchers with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("FastTextSuite")
      .getOrCreate()
  }

  override protected def afterAll(): Unit = {
    try {
      spark.sparkContext.stop()
    } finally {
      super.afterAll()
    }
  }

  //Test case for the scenario mentioned in the paper
  test("computeSubWords") {
    val word = "where"
    val index = 0
    val vocabSize = 1
    val minn = 3
    val maxn = 3
    val bucket = 2000000
    val ft = new FastText().setMinn(minn).setMaxn(maxn).setBucket(bucket)
    val computeSubwords = PrivateMethod[Array[Int]]('computeSubwords)
    val hash = PrivateMethod[BigInt]('hash)
    val subwords = ft invokePrivate computeSubwords(word, vocabSize, index)
    val truthLabelsStr = Array("<where>", "<wh", "whe", "her", "ere", "re>")
    val truthLabelsHash = truthLabelsStr.map(s => {
      if (s.equals("<where>"))
        0
      else
        vocabSize + ((ft invokePrivate hash(s)) mod bucket).intValue
    })
    subwords.foreach(h => assert(truthLabelsHash.contains(h)))
  }

  test("findSynonyms") {
    val sentence = "a b " * 100 + "a c " * 10
    val localDoc = Seq(sentence, sentence)
    val doc = spark.sparkContext.parallelize(localDoc).map(line => line.split(" ").toSeq)
    val model = new FastText().setVectorSize(10).setSeed(42L).fit(doc)
    val syms = model.findSynonyms("a", 2)
    assert(syms.length == 2)
    assert(syms(0)._1 == "b")
    assert(syms(1)._1 == "c")
  }

  test("FastText throws exception when vocabulary is empty") {
    intercept[IllegalArgumentException] {
      val sentence = "a b c"
      val localDoc = Seq(sentence, sentence)
      val doc = spark.sparkContext.parallelize(localDoc)
        .map(line => line.split(" ").toSeq)
      new FastText().setMinCount(10).fit(doc)
    }
  }
}
