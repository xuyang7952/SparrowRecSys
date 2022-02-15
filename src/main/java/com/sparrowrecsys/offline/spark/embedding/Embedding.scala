package com.sparrowrecsys.offline.spark.embedding

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object Embedding {

  val redisEndpoint = "localhost"
  val redisPort = 6379

  def processItemSequence(sparkSession: SparkSession, rawSampleDataPath: String): RDD[Seq[String]] = {
    // 将原始数据转成spark中的rdd
    //path of rating data
    //设定rating数据的路径并用spark载入数据
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    //sort by timestamp udf
    //实现一个用户定义的操作函数(UDF)，用于之后的排序
    val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
      rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
        .sortBy { case (_, timestamp) => timestamp }
        .map { case (movieId, _) => movieId }
    })

    ratingSamples.printSchema()

    //process rating data then generate rating movie sequence data
    //把原始的rating数据处理成序列数据，
    val userSeq = ratingSamples
      .where(col("rating") >= 3.5) //过滤掉评分在3.5一下的评分记录
      .groupBy("userId") //按照用户id分组
      .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds") //每个用户生成一个序列并用刚才定义好的udf函数按照timestamp排序
      .withColumn("movieIdStr", array_join(col("movieIds"), " ")) //把所有id连接成一个String，方便后续word2vec模型处理

    //把序列数据筛选出来，丢掉其他过程数据
    userSeq.select("userId", "movieIdStr").show(10, truncate = false) // 展示部分数据
    userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
  }

  def generateUserEmb(sparkSession: SparkSession, rawSampleDataPath: String, word2VecModel: Word2VecModel, embLength: Int, embOutputFilename: String, saveToRedis: Boolean, redisKeyPrefix: String): Unit = {
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    println("ratingSamples--评分数据：")
    ratingSamples.show(10, false)

    val userEmbeddings = new ArrayBuffer[(String, Array[Float])]()
    // 将用户看过的电影的item的Embedding的平均值，当做用户Embedding
    ratingSamples.collect().groupBy(_.getAs[String]("userId"))
      .foreach(user => {
        val userId = user._1
        var userEmb = new Array[Float](embLength)

        var movieCount = 0
        userEmb = user._2.foldRight[Array[Float]](userEmb)((row, newEmb) => {
          val movieId = row.getAs[String]("movieId")
          val movieEmb = word2VecModel.getVectors.get(movieId)
          movieCount += 1
          if (movieEmb.isDefined) {
            newEmb.zip(movieEmb.get).map { case (x, y) => x + y }
          } else {
            newEmb
          }
        }).map((x: Float) => x / movieCount)
        userEmbeddings.append((userId, userEmb))
      })


    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))

    for (userEmb <- userEmbeddings) {
      bw.write(userEmb._1 + ":" + userEmb._2.mkString(" ") + "\n")
    }
    bw.close()

    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //set ttl to 24hs
      params.ex(60 * 60 * 24)

      for (userEmb <- userEmbeddings) {
        redisClient.set(redisKeyPrefix + ":" + userEmb._1, userEmb._2.mkString(" "), params)
      }
      redisClient.close()
    }
  }

  def trainItem2vec(sparkSession: SparkSession, samples: RDD[Seq[String]], embLength: Int, embOutputFilename: String, saveToRedis: Boolean, redisKeyPrefix: String): Word2VecModel = {
    //设置模型参数，Embedding向量维度数量，Word2Vec滑动窗口大小，迭代次数
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)
      .setWindowSize(5)
      .setNumIterations(10)

    //训练模型
    val model = word2vec.fit(samples)

    //训练结束，用模型查找与item"592"最相似的20个item
    println("用模型查找与item'158'最相似的5个item")
    val synonyms = model.findSynonyms("158", 5)
    for ((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    // 保存模型
    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))
    //用model.getVectors获取所有Embedding向量
    for (movieId <- model.getVectors.keys) {
      bw.write(movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
    }
    bw.close()

    // 保存到redis里
    if (saveToRedis) {
      //创建redis client
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //set ttl to 24hs
      //设置ttl为24小时
      params.ex(60 * 60 * 24)
      //遍历存储embedding向量
      for (movieId <- model.getVectors.keys) {
        //key的形式为前缀+movieId，例如i2vEmb:361
        //value的形式是由Embedding向量生成的字符串，例如 "0.1693846 0.2964318 -0.13044095 0.37574086 0.55175656 0.03217995 1.327348 -0.81346786 0.45146862 0.49406642"
        redisClient.set(redisKeyPrefix + ":" + movieId, model.getVectors(movieId).mkString(" "), params)
      }
      //关闭客户端连接
      redisClient.close()
    }

    embeddingLSH(sparkSession, model.getVectors)
    model
  }

  /**
   * //通过随机游走产生一个样本的过程
   * transferMatrix 转移概率矩阵//itemCount 物品出现次数的分布//itemTotalCount 物品出现总次数//sampleLength 每个样本的长度
   */
  def oneRandomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleLength: Int): Seq[String] = {
    val sample = mutable.ListBuffer[String]()

    //pick the first element
    //决定起始点
    val randomDouble = Random.nextDouble()
    var firstItem = ""
    var accumulateProb: Double = 0D
    //根据物品出现的概率，随机决定起始点
    breakable {
      for ((item, prob) <- itemDistribution) {
        accumulateProb += prob
        if (accumulateProb >= randomDouble) {
          firstItem = item
          break
        }
      }
    }

    sample.append(firstItem)
    var curElement = firstItem
    //通过随机游走产生长度为sampleLength的样本
    breakable {
      for (_ <- 1 until sampleLength) {
        if (!itemDistribution.contains(curElement) || !transitionMatrix.contains(curElement)) {
          break
        }

        //从curElement到下一个跳的转移概率向量
        val probDistribution = transitionMatrix(curElement)
        val randomDouble = Random.nextDouble()
        var accumulateProb: Double = 0D
        //根据转移概率向量随机决定下一跳的物品
        breakable {
          for ((item, prob) <- probDistribution) {
            accumulateProb += prob
            if (accumulateProb >= randomDouble) {
              curElement = item
              break
            }
          }
        }
        sample.append(curElement)
      }
    }
    Seq(sample.toList: _*)
  }

  /*
  //随机游走采样函数//transferMatrix 转移概率矩阵//itemCount 物品出现次数的分布
   */
  def randomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleCount: Int, sampleLength: Int): Seq[Seq[String]] = {

    val samples = mutable.ListBuffer[Seq[String]]()
    //随机游走sampleCount次，生成sampleCount个序列样本
    for (_ <- 1 to sampleCount) {
      samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    }
    Seq(samples.toList: _*)
  }

  def generateTransitionMatrix(samples: RDD[Seq[String]]): (mutable.Map[String, mutable.Map[String, Double]], mutable.Map[String, Double]) = {
    //Graph Embedding的Deep Walk 方法的gaily转移矩阵
    //samples 输入的观影序列样本集，通过flatMap操作把观影序列打碎成一个个影片对
    val pairSamples = samples.flatMap[(String, String)](sample => {
      var pairSeq = Seq[(String, String)]()
      var previousItem: String = null
      sample.foreach((element: String) => {
        if (previousItem != null) {
          pairSeq = pairSeq :+ (previousItem, element)
        }
        previousItem = element
      })
      pairSeq
    })

    //统计影片对的数量
    val pairCountMap = pairSamples.countByValue()
    var pairTotalCount = 0L
    //转移概率矩阵的双层Map数据结构
    val transitionCountMatrix = mutable.Map[String, mutable.Map[String, Long]]()
    val itemCountMap = mutable.Map[String, Long]()

    //求取转移概率矩阵
    pairCountMap.foreach(pair => {
      val pairItems = pair._1
      val count = pair._2

      if (!transitionCountMatrix.contains(pairItems._1)) {
        transitionCountMatrix(pairItems._1) = mutable.Map[String, Long]()
      }

      transitionCountMatrix(pairItems._1)(pairItems._2) = count
      itemCountMap(pairItems._1) = itemCountMap.getOrElse[Long](pairItems._1, 0) + count
      pairTotalCount = pairTotalCount + count
    })

    val transitionMatrix = mutable.Map[String, mutable.Map[String, Double]]()
    val itemDistribution = mutable.Map[String, Double]()

    transitionCountMatrix foreach {
      case (itemAId, transitionMap) =>
        transitionMatrix(itemAId) = mutable.Map[String, Double]()
        transitionMap foreach { case (itemBId, transitionCount) => transitionMatrix(itemAId)(itemBId) = transitionCount.toDouble / itemCountMap(itemAId) }
    }

    itemCountMap foreach { case (itemId, itemCount) => itemDistribution(itemId) = itemCount.toDouble / pairTotalCount }
    (transitionMatrix, itemDistribution)
  }

  /**
   * 局部敏感哈希算法的实现
   * @param spark
   * @param movieEmbMap
   */
  def embeddingLSH(spark: SparkSession, movieEmbMap: Map[String, Array[Float]]): Unit = {
    //将电影embedding数据转换成dense Vector的形式，便于之后处理
    val movieEmbSeq = movieEmbMap.toSeq.map(item => (item._1, Vectors.dense(item._2.map(f => f.toDouble))))
    val movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")

    //LSH bucket model
    //利用Spark MLlib创建LSH分桶模型,BucketLength 指的就是分桶公式中的分桶宽度w，NumHashTables 指的是哈希表，函数的数量
    val bucketProjectionLSH = new BucketedRandomProjectionLSH()
      .setBucketLength(0.1)//根据hash函数计算出来的值，用0.1的宽度进行划分分桶，hash值的范围应该是-1~1.
      .setNumHashTables(3)// 有三个hash函数，每个函数会计算出来一定范围的值
      .setInputCol("emb")
      .setOutputCol("bucketId")

    //训练LSH分桶模型
    val bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    //进行分桶
    val embBucketResult = bucketModel.transform(movieEmbDF)
    println("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    println("movieId, emb, bucketId data 局部敏感哈希LSH分桶结果 result:")
    embBucketResult.show(10, truncate = false)

    //尝试对一个示例Embedding查找最近邻
    println("Approximately searching for 5 nearest neighbors of the sample embedding:")
    val sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate = false)
  }

  def graphEmb(samples: RDD[Seq[String]], sparkSession: SparkSession, embLength: Int, embOutputFilename: String, saveToRedis: Boolean, redisKeyPrefix: String): Word2VecModel = {
    // 图结构数据，生成物品Embedding,Graph Embedding：随机游走采样过程
    val transitionMatrixAndItemDis = generateTransitionMatrix(samples)

    println("graphEmb--transitionMatrixAndItemDis的相关信息：")
    println(transitionMatrixAndItemDis._1.size)
    println(transitionMatrixAndItemDis._2.size)

    //样本的数量//每个样本的长度
    val sampleCount = 20000
    val sampleLength = 10
    val newSamples = randomWalk(transitionMatrixAndItemDis._1, transitionMatrixAndItemDis._2, sampleCount, sampleLength)

    val rddSamples = sparkSession.sparkContext.parallelize(newSamples)
    // 利用得到的序列数据,trainItem2vec方法，生成Embedding向量
    trainItem2vec(sparkSession, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val rawSampleDataPath = "/webroot/sampledata/ratings.csv"
    val embLength = 10

    // 读取文件并处理，
    println("----------读取文件并处理----------")
    val samples = processItemSequence(spark, rawSampleDataPath)
    println("samples.first():", samples.first())
    println("----------利用item序列信息转成物品Embedding向量----------")
    val model = trainItem2vec(spark, samples, embLength, "item2vecEmb.csv", saveToRedis = true, "i2vEmb")
    println("----------利用图数据信息转成物品Embedding向量----------")
    graphEmb(samples, spark, embLength, "itemGraphEmb.csv", saveToRedis = true, "graphEmb")
    println("----------利用物品Embedding向量生成用户Embedding向量----------")
    generateUserEmb(spark, rawSampleDataPath, model, embLength, "userEmb.csv", saveToRedis = false, "uEmb")
  }
}
