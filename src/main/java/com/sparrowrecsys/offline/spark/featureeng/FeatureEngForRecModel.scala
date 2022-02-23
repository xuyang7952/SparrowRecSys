package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions.{format_number, _}
import org.apache.spark.sql.types.{DecimalType, FloatType, IntegerType, LongType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.immutable.ListMap
import scala.collection.{JavaConversions, mutable}

object FeatureEngForRecModel {

  val NUMBER_PRECISION = 2
  val redisEndpoint = "localhost"
  val redisPort = 6379

  /*
   给评分数据集添加label，大于3.5分置为1，小于3.5的置为0，模拟ctr预估
   */
  def addSampleLabel(ratingSamples: DataFrame): DataFrame = {
    println("addSampleLabel---ratingSamples的前5行数据：")
    ratingSamples.show(5, truncate = false)
    ratingSamples.printSchema()
    val sampleCount = ratingSamples.count()
    // 计算评分的分布，
    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
      .withColumn("percentage", col("count") / sampleCount).show(10, truncate = false)

    ratingSamples.withColumn("label", when(col("rating") >= 3.5, 1).otherwise(0))
  }

  /*
  添加电影特征
   */
  def addMovieFeatures(movieSamples: DataFrame, ratingSamples: DataFrame): DataFrame = {

    //add movie basic features
    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")
    //add release year--发布年份，默认1990
    val extractReleaseYearUdf = udf({ (title: String) => {
      if (null == title || title.trim.length < 6) {
        1990 // default value
      }
      else {
        val yearString = title.trim.substring(title.length - 5, title.length - 1)
        yearString.toInt
      }
    }
    })

    //add title--添加标题
    val extractTitleUdf = udf({ (title: String) => {
      title.trim.substring(0, title.trim.length - 6).trim
    }
    })

    val samplesWithMovies2 = samplesWithMovies1.withColumn("releaseYear", extractReleaseYearUdf(col("title")))
      .withColumn("title", extractTitleUdf(col("title")))
      .drop("title") //title is useless currently

    //split genres---切分电影风格标签，取前三个风格
    val samplesWithMovies3 = samplesWithMovies2.withColumn("movieGenre1", split(col("genres"), "\\|").getItem(0))
      .withColumn("movieGenre2", split(col("genres"), "\\|").getItem(1))
      .withColumn("movieGenre3", split(col("genres"), "\\|").getItem(2))

    //add rating features---添加电影评分特征，评分count，平均分，标准差
    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
      .agg(count(lit(1)).as("movieRatingCount"),
        format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
        stddev(col("rating")).as("movieRatingStddev"))
      .na.fill(0).withColumn("movieRatingStddev", format_number(col("movieRatingStddev"), NUMBER_PRECISION))


    //join movie rating features
    val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieId"), "left")
    println("addMovieFeatures,根据评分数据，和电影数据，提炼出的特征")
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(5, truncate = false)

    samplesWithMovies4
  }

  val extractGenres: UserDefinedFunction = udf { (genreArray: Seq[String]) => {
    val genreMap = mutable.Map[String, Int]()
    genreArray.foreach((element: String) => {
      val genres = element.split("\\|")
      genres.foreach((oneGenre: String) => {
        genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0) + 1
      })
    })
    val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2): _*)
    sortedGenres.keys.toSeq
  }
  }

  /*
  添加用户特征
   */
  def addUserFeatures(ratingSamples: DataFrame): DataFrame = {
    val samplesWithUserFeatures = ratingSamples
      .withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieId")).otherwise(lit(null)))
        .over(Window.partitionBy("userId") // 防止引入未来信息
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userPositiveHistory", reverse(col("userPositiveHistory"))) //用户评分大于3.5的movieid的集合
      .withColumn("userRatedMovie1", col("userPositiveHistory").getItem(0)) //最近5部喜欢的电影
      .withColumn("userRatedMovie2", col("userPositiveHistory").getItem(1))
      .withColumn("userRatedMovie3", col("userPositiveHistory").getItem(2))
      .withColumn("userRatedMovie4", col("userPositiveHistory").getItem(3))
      .withColumn("userRatedMovie5", col("userPositiveHistory").getItem(4))
      .withColumn("userRatingCount", count(lit(1)) //总评分次数
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgReleaseYear", avg(col("releaseYear")) //喜欢电影平均发布时间
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)).cast(IntegerType))
      .withColumn("userReleaseYearStddev", stddev(col("releaseYear")) //喜欢电影发布年份标准差
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", format_number(avg(col("rating")) //喜欢电影平均评分
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userRatingStddev", stddev(col("rating")) //喜欢电影平均评分标准差
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userGenres", extractGenres(collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
        .over(Window.partitionBy("userId") //喜欢电影类型
          .orderBy(col("timestamp")).rowsBetween(-100, -1))))
      .na.fill(0)
      .withColumn("userRatingStddev", format_number(col("userRatingStddev"), NUMBER_PRECISION))
      .withColumn("userReleaseYearStddev", format_number(col("userReleaseYearStddev"), NUMBER_PRECISION))
      .withColumn("userGenre1", col("userGenres").getItem(0)) //用户喜欢的5种风格
      .withColumn("userGenre2", col("userGenres").getItem(1))
      .withColumn("userGenre3", col("userGenres").getItem(2))
      .withColumn("userGenre4", col("userGenres").getItem(3))
      .withColumn("userGenre5", col("userGenres").getItem(4))
      .drop("genres", "userGenres", "userPositiveHistory") //删除中间特征
      .filter(col("userRatingCount") > 1) //删除没有评分的数据

    println("samplesWithUserFeatures--添加后的用户特征")
    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(10, truncate = false)

    samplesWithUserFeatures
  }

  /*
  保存电影特征数据到redis
   */
  def extractAndSaveMovieFeaturesToRedis(samples: DataFrame): DataFrame = {
    val movieLatestSamples = samples.withColumn("movieRowNum", row_number()
      .over(Window.partitionBy("movieId")
        .orderBy(col("timestamp").desc)))
      .filter(col("movieRowNum") === 1)
      .select("movieId", "releaseYear", "movieGenre1", "movieGenre2", "movieGenre3", "movieRatingCount",
        "movieAvgRating", "movieRatingStddev")
      .na.fill("")

    println("movieLatestSamples--需要保存到redis里的电影特征")
    movieLatestSamples.printSchema()
    movieLatestSamples.show(10, truncate = false)

    val movieFeaturePrefix = "mf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    val sampleArray = movieLatestSamples.collect()
    println("total movie size:" + sampleArray.length)
    var insertedMovieNumber = 0
    val movieCount = sampleArray.length
    for (sample <- sampleArray) {
      val movieKey = movieFeaturePrefix + sample.getAs[String]("movieId")
      val valueMap = mutable.Map[String, String]()
      valueMap("movieGenre1") = sample.getAs[String]("movieGenre1")
      valueMap("movieGenre2") = sample.getAs[String]("movieGenre2")
      valueMap("movieGenre3") = sample.getAs[String]("movieGenre3")
      valueMap("movieRatingCount") = sample.getAs[Long]("movieRatingCount").toString
      valueMap("releaseYear") = sample.getAs[Int]("releaseYear").toString
      valueMap("movieAvgRating") = sample.getAs[String]("movieAvgRating")
      valueMap("movieRatingStddev") = sample.getAs[String]("movieRatingStddev")

      redisClient.hset(movieKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedMovieNumber += 1
      if (insertedMovieNumber % 100 == 0) {
        println(insertedMovieNumber + "/" + movieCount + "...")
      }
    }

    redisClient.close()
    movieLatestSamples
  }

  /*
  分割处理后特征数据，保存为train，test数据
   */
  def splitAndSaveTrainingTestSamples(samples: DataFrame, savePath: String) = {
    //generate a smaller sample set for demo
    val smallSamples = samples.sample(0.1) //抽样

    //split training and test set by 8:2，随机切分数据
    val Array(training, test) = smallSamples.randomSplit(Array(0.8, 0.2))

    val sampleResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath + "/trainingSamples")
    test.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath + "/testSamples")
  }

  /*
  根据时间，分割样本
   */
  def splitAndSaveTrainingTestSamplesByTimeStamp(samples: DataFrame, savePath: String) = {
    //generate a smaller sample set for demo
    val smallSamples = samples.sample(0.1).withColumn("timestampLong", col("timestamp").cast(LongType))

    //找到时间切割点
    val quantile = smallSamples.stat.approxQuantile("timestampLong", Array(0.8), 0.05)
    val splitTimestamp = quantile.apply(0)

    //切割样本为训练集和测试集
    val training = smallSamples.where(col("timestampLong") <= splitTimestamp).drop("timestampLong")
    val test = smallSamples.where(col("timestampLong") > splitTimestamp).drop("timestampLong")

    val sampleResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath + "/trainingSamples")
    test.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath + "/testSamples")
  }

  /*
  保存用户特征数据到redis
   */
  def extractAndSaveUserFeaturesToRedis(samples: DataFrame): DataFrame = {
    val userLatestSamples = samples.withColumn("userRowNum", row_number()
      .over(Window.partitionBy("userId")
        .orderBy(col("timestamp").desc)))
      .filter(col("userRowNum") === 1)
      .select("userId", "userRatedMovie1", "userRatedMovie2", "userRatedMovie3", "userRatedMovie4", "userRatedMovie5",
        "userRatingCount", "userAvgReleaseYear", "userReleaseYearStddev", "userAvgRating", "userRatingStddev",
        "userGenre1", "userGenre2", "userGenre3", "userGenre4", "userGenre5")
      .na.fill("")

    println("userLatestSamples---需要保存到redis的用户特征数据：")
    userLatestSamples.printSchema()
    userLatestSamples.show(10, truncate = false)

//    val userFeaturePrefix = "uf:"

//    val redisClient = new Jedis(redisEndpoint, redisPort)
//    val params = SetParams.setParams()
//    //set ttl to 24hs * 30
//    params.ex(60 * 60 * 24 * 30)
//    val sampleArray = userLatestSamples.collect()
//    println("total user size:" + sampleArray.length)
//    var insertedUserNumber = 0
//    val userCount = sampleArray.length
//    for (sample <- sampleArray) {
//      val userKey = userFeaturePrefix + sample.getAs[String]("userId")
//      val valueMap = mutable.Map[String, String]()
//      valueMap("userRatedMovie1") = sample.getAs[String]("userRatedMovie1")
//      valueMap("userRatedMovie2") = sample.getAs[String]("userRatedMovie2")
//      valueMap("userRatedMovie3") = sample.getAs[String]("userRatedMovie3")
//      valueMap("userRatedMovie4") = sample.getAs[String]("userRatedMovie4")
//      valueMap("userRatedMovie5") = sample.getAs[String]("userRatedMovie5")
//      valueMap("userGenre1") = sample.getAs[String]("userGenre1")
//      valueMap("userGenre2") = sample.getAs[String]("userGenre2")
//      valueMap("userGenre3") = sample.getAs[String]("userGenre3")
//      valueMap("userGenre4") = sample.getAs[String]("userGenre4")
//      valueMap("userGenre5") = sample.getAs[String]("userGenre5")
//      valueMap("userRatingCount") = sample.getAs[Long]("userRatingCount").toString
//      valueMap("userAvgReleaseYear") = sample.getAs[Int]("userAvgReleaseYear").toString
//      valueMap("userReleaseYearStddev") = sample.getAs[String]("userReleaseYearStddev")
//      valueMap("userAvgRating") = sample.getAs[String]("userAvgRating")
//      valueMap("userRatingStddev") = sample.getAs[String]("userRatingStddev")
//
//      redisClient.hset(userKey, JavaConversions.mapAsJavaMap(valueMap))
//      insertedUserNumber += 1
//      if (insertedUserNumber % 100 == 0) {
//        println(insertedUserNumber + "/" + userCount + "...")
//      }
//    }
//
//    redisClient.close()
    userLatestSamples
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    println("-----------读取电影本身信息数据------------")
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    println("movieSamples前5行数据", movieSamples.show(5))

    println("-----------读取电影评分信息数据------------")
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    println("ratingSamples前5行数据：", ratingSamples.show(5))

    //添加样本标签
    val ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    println("-----------ratingSamplesWithLabel前5行数据------------")
    ratingSamplesWithLabel.show(5, truncate = false)

    //添加物品（电影）特征
    val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    //添加用户特征
    val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)


    //save samples as csv format--以csv格式保存数据
//    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, "/webroot/sampledata")

    //save user features and item features to redis for online inference
    extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
    extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)
    spark.close()
  }

}
