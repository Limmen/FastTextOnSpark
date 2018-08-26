package com.github.limmen.spark.mllib.fasttext

import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.PrivateMethodTester._
import org.scalatest._

class FastTextSuite extends FunSuite with Matchers with BeforeAndAfterAll {

  private var spark: SparkSession = _

  /**
   * Test data
   */
  val wordsModelTest = Map(
    ("china", Array(0.50f, 0.50f, 0.50f, 0.50f)),
    ("japan", Array(0.40f, 0.50f, 0.50f, 0.50f)),
    ("taiwan", Array(0.60f, 0.50f, 0.50f, 0.50f)),
    ("korea", Array(0.45f, 0.60f, 0.60f, 0.60f)))
  val subwordsModelTest = Map(
    ("<ch", Array(0.51f, 0.50f, 0.50f, 0.50f)),
    ("chi", Array(0.52f, 0.50f, 0.50f, 0.50f)),
    ("hin", Array(0.53f, 0.50f, 0.50f, 0.50f)),
    ("ina", Array(0.54f, 0.50f, 0.50f, 0.50f)),
    ("na>", Array(0.55f, 0.50f, 0.50f, 0.50f)),
    ("<ja", Array(0.41f, 0.50f, 0.50f, 0.50f)),
    ("jap", Array(0.42f, 0.50f, 0.50f, 0.50f)),
    ("apa", Array(0.43f, 0.50f, 0.50f, 0.50f)),
    ("pan", Array(0.44f, 0.50f, 0.50f, 0.50f)),
    ("an>", Array(0.45f, 0.50f, 0.50f, 0.50f)),
    ("<ta", Array(0.61f, 0.50f, 0.50f, 0.50f)),
    ("tai", Array(0.62f, 0.50f, 0.50f, 0.50f)),
    ("aiw", Array(0.63f, 0.50f, 0.50f, 0.50f)),
    ("iwa", Array(0.64f, 0.50f, 0.50f, 0.50f)),
    ("wan", Array(0.65f, 0.50f, 0.50f, 0.50f)),
    ("an>", Array(0.65f, 0.50f, 0.50f, 0.50f)),
    ("<ko", Array(0.71f, 0.60f, 0.60f, 0.60f)),
    ("kor", Array(0.72f, 0.60f, 0.60f, 0.60f)),
    ("ore", Array(0.73f, 0.60f, 0.60f, 0.60f)),
    ("rea", Array(0.74f, 0.60f, 0.60f, 0.60f)),
    ("ea>", Array(0.75f, 0.60f, 0.60f, 0.60f)))

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

  //Test case for the scenario mentioned in the FastText paper for computing subwords
  test("computeSubWords") {
    val word = "where"
    val index = 0
    val vocabSize = 1
    val minn = 3
    val maxn = 3
    val bucket = 2000000
    val ft = new FastText().setMinn(minn).setMaxn(maxn).setBucket(bucket)
    val computeSubwords = PrivateMethod[Array[Int]]('computeSubwords)
    val subwords = ft invokePrivate computeSubwords(word, vocabSize, index)
    val truthLabelsStr = Array("<where>", "<wh", "whe", "her", "ere", "re>")
    val truthLabelsHash = truthLabelsStr.map(s => {
      if (s.equals("<where>"))
        0
      else
        vocabSize + ((FastText.hash(s)) mod bucket).intValue
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

  test("buildWordVectorsFromMaps") {
    val sentence = "china japan taiwan korea"
    val localDoc = Seq(sentence, sentence)
    val doc = spark.sparkContext.parallelize(localDoc).map(line => line.split(" ").toSeq)
    val bucket = 1000
    val vectorSize = 4
    val vocab = new FastText().setVectorSize(vectorSize).setSeed(42L).setMinn(3).setMaxn(3).setBucket(bucket).setMinCount(1).fit(doc).vocab
    val vocabSize = vocab.size
    println(s"vocabSize: ${vocab.size}")
    val wordVectors = FastTextModel.buildWordVectors(wordsModelTest, subwordsModelTest, vocab, bucket)
    assert(wordVectors.slice(0, vectorSize).sameElements(wordsModelTest("china")))
    assert(wordVectors.slice(vectorSize, 2 * vectorSize).sameElements(wordsModelTest("japan")))
    assert(wordVectors.slice(2 * vectorSize, 3 * vectorSize).sameElements(wordsModelTest("taiwan")))
    assert(wordVectors.slice(3 * vectorSize, 4 * vectorSize).sameElements(wordsModelTest("korea")))

    var hash = (vocabSize + (FastText.hash("<ch") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("<ch")))
    hash = (vocabSize + (FastText.hash("chi") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("chi")))
    hash = (vocabSize + (FastText.hash("hin") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("hin")))
    hash = (vocabSize + (FastText.hash("ina") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("ina")))
    hash = (vocabSize + (FastText.hash("na>") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("na>")))
    hash = (vocabSize + (FastText.hash("<ja") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("<ja")))
    hash = (vocabSize + (FastText.hash("jap") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("jap")))
    hash = (vocabSize + (FastText.hash("apa") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("apa")))
    hash = (vocabSize + (FastText.hash("pan") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("pan")))
    hash = (vocabSize + (FastText.hash("an>") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("an>")))
    hash = (vocabSize + (FastText.hash("<ta") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("<ta")))
    hash = (vocabSize + (FastText.hash("tai") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("tai")))
    hash = (vocabSize + (FastText.hash("aiw") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("aiw")))
    hash = (vocabSize + (FastText.hash("iwa") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("iwa")))
    hash = (vocabSize + (FastText.hash("wan") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("wan")))
    hash = (vocabSize + (FastText.hash("an>") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("an>")))
    hash = (vocabSize + (FastText.hash("<ko") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("<ko")))
    hash = (vocabSize + (FastText.hash("kor") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("kor")))
    hash = (vocabSize + (FastText.hash("ore") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("ore")))
    hash = (vocabSize + (FastText.hash("rea") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("rea")))
    hash = (vocabSize + (FastText.hash("ea>") mod bucket).intValue)
    assert(wordVectors.slice(hash * vectorSize, hash * vectorSize + vectorSize).sameElements(subwordsModelTest("ea>")))

    assert(wordVectors.length == vectorSize * vocabSize + bucket * vectorSize)
  }

  test("buildModelFromMaps") {
    val sentence = "china japan taiwan korea"
    val localDoc = Seq(sentence, sentence)
    val doc = spark.sparkContext.parallelize(localDoc).map(line => line.split(" ").toSeq)
    val num = 2
    val bucket = 1000
    val vectorSize = 4
    val vocab = new FastText().setVectorSize(vectorSize).setSeed(42L).setMinn(3).setMaxn(3).setBucket(bucket).setMinCount(1).fit(doc).vocab
    val model = new FastTextModel(wordsModelTest, subwordsModelTest, vocab, bucket)
    val syms = model.findSynonyms("china", num)
    assert(syms.length == num)
    assert(syms(0)._1 == "taiwan")
    assert(syms(1)._1 == "japan")
  }

  test("findSynonyms doesn't reject similar word vectors when called with a vector") {
    val sentence = "china japan taiwan korea"
    val localDoc = Seq(sentence, sentence)
    val doc = spark.sparkContext.parallelize(localDoc).map(line => line.split(" ").toSeq)
    val num = 2
    val bucket = 1000
    val vectorSize = 4
    val vocab = new FastText().setVectorSize(vectorSize).setSeed(42L).setMinn(3).setMaxn(3).setBucket(bucket).setMinCount(1).fit(doc).vocab
    val model = new FastTextModel(wordsModelTest, subwordsModelTest, vocab, bucket)
    val syms = model.findSynonyms(Vectors.dense(Array(0.52, 0.5, 0.5, 0.5)), num)
    assert(syms.length == num)
    assert(syms(0)._1 == "china")
    assert(syms(1)._1 == "taiwan")
  }

  test("test similarity for word vectors with large values is not Infinity or NaN") {
    val vecA = Array(-4.331467827487745E21, -5.26707742075006E21,
      5.63551690626524E21, 2.833692188614257E21, -1.9688159903619345E21, -4.933950659913092E21,
      -2.7401535502536787E21, -1.418671793782632E20).map(_.toFloat)
    val vecB = Array(-3.9850175451103232E16, -3.4829783883841536E16,
      9.421469251534848E15, 4.4069684466679808E16, 7.20936298872832E15, -4.2883302830374912E16,
      -3.605579947835392E16, -2.8151294422155264E16).map(_.toFloat)
    val vecC = Array(-1.9227381025734656E16, -3.907009342603264E16,
      2.110207626838016E15, -4.8770066610651136E16, -1.9734964555743232E16, -3.2206001247617024E16,
      2.7725358220443648E16, 3.1618718156980224E16).map(_.toFloat)
    val wordsModel = Map(
      ("A", vecA),
      ("B", vecB),
      ("C", vecC))
    val subwordsModel = Map[String, Array[Float]]()
    val sentence = "A B C"
    val localDoc = Seq(sentence, sentence)
    val doc = spark.sparkContext.parallelize(localDoc).map(line => line.split(" ").toSeq)
    val bucket = 1000
    val vectorSize = 4
    val vocab = new FastText().setVectorSize(vectorSize).setSeed(42L).setMinn(3).setMaxn(3).setBucket(bucket).setMinCount(1).fit(doc).vocab
    val model = new FastTextModel(wordsModel, subwordsModel, vocab, bucket)
    model.findSynonyms("A", 5).foreach { pair =>
      assert(!(pair._2.isInfinite || pair._2.isNaN))
    }
  }

  test("wordVecNorms") {
    val testVector1 = Array(0.6f, -0.2f, 0.9f, 1.2f, -2.4f)
    val vocabSize1 = 1
    val vectorSize1 = 5
    val bucket1 = 0
    val norm1 = FastTextModel.wordVecNorms(testVector1, vocabSize1, vectorSize1, bucket1)
    assert(norm1.length == 1)
    assert(norm1(0) == 2.9f)
    val testVector2 = Array(0.6f, -0.2f, 0.9f, 1.2f, -2.4f, 1.39f, -1.56f, 0.11f, 0.165f, -0.789f)
    val vocabSize2 = 2
    val norm2 = FastTextModel.wordVecNorms(testVector2, vocabSize2, vectorSize1, bucket1)
    assert(norm2.length == 2)
    assert(norm2(0) == 2.9f)
    assert(norm2(1) == 2.242219f)
  }

  test("normalizeVecs") {
    val testVector1 = Array(0.6f, -0.2f, 0.9f, 1.2f, -2.4f)
    val vocabSize1 = 1
    val vectorSize1 = 5
    val bucket1 = 0
    val norm1 = FastTextModel.wordVecNorms(testVector1, vocabSize1, vectorSize1, bucket1)
    val normalizedVector1 = FastTextModel.normalizeVecs(testVector1, norm1, vocabSize1, vectorSize1, bucket1)
    assert(normalizedVector1.length == testVector1.length)
    assert(normalizedVector1.sameElements(Array(0.20689654f, -0.06896552f, 0.3103448f, 0.4137931f, -0.8275862f)))
    val testVector2 = Array(0.6f, -0.2f, 0.9f, 1.2f, -2.4f, 1.39f, -1.56f, 0.11f, 0.165f, -0.789f)
    val vocabSize2 = 2
    val norm2 = FastTextModel.wordVecNorms(testVector2, vocabSize2, vectorSize1, bucket1)
    val normalizedVector2 = FastTextModel.normalizeVecs(testVector2, norm2, vocabSize2, vectorSize1, bucket1)
    normalizedVector2.foreach(println)
    assert(normalizedVector2.length == testVector2.length)
    assert(normalizedVector2.sameElements(Array(
      0.20689654f, -0.06896552f, 0.3103448f, 0.4137931f, -0.8275862f,
      0.6199216f, -0.6957393f, 0.049058545f, 0.07358782f, -0.35188356f)))
  }

  test("transform") {
    val sentence = "china japan taiwan korea"
    val localDoc = Seq(sentence, sentence)
    val doc = spark.sparkContext.parallelize(localDoc).map(line => line.split(" ").toSeq)
    val bucket = 1000
    val vectorSize = 4
    val vocab = new FastText().setVectorSize(vectorSize).setSeed(42L).setMinn(3).setMaxn(3).setBucket(bucket).setMinCount(1).fit(doc).vocab
    val model = new FastTextModel(wordsModelTest, subwordsModelTest, vocab, bucket)
    assert(model.transform("china").toArray.sameElements(wordsModelTest("china")))
    assert(model.transform("japan").toArray.sameElements(wordsModelTest("japan")))
    assert(model.transform("taiwan").toArray.sameElements(wordsModelTest("taiwan")))
    assert(model.transform("korea").toArray.sameElements(wordsModelTest("korea")))
  }

  //  test("getVectors") {
  //    ???
  //  }
  //
  //
  //
  //  test("window size") {
  //    ???
  //  }
}
