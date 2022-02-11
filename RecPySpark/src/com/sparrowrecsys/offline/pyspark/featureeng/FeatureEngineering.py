from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


def oneHotEncoderExample(movieSamples):
    """独热编码的案例--对movieId进行独热编码"""
    # 使用WithColumn可以为DataFrame增加新列
    # cast强制类型转换
    samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))
    # 创建onehot编码器
    encoder = OneHotEncoder(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=False)
    # 训练onehot编码器，，transform转换
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    # 打印数据结构
    oneHotEncoderSamples.printSchema()
    oneHotEncoderSamples.show(10)


def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list)


def multiHotEncoderExample(movieSamples):
    """多热编码的案例"""
    samplesWithGenre = movieSamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndexInt",
                                                                                  F.col("genreIndex").cast(IntegerType()))
    indexSize = genreIndexSamples.agg(max(F.col("genreIndexInt"))).head()[0] + 1
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(indexSize))
    finalSample = processedSamples.withColumn("vector",
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    finalSample.printSchema()
    try:
        finalSample.show(10)  # 数据太多，无法显示？
    except Exception as e:
        print(f"Exception:{e}")


def ratingFeatures(ratingSamples):
    """评分数据处理，数值类型，归一化和分桶"""
    ratingSamples.printSchema()
    # ratingSamples.show(10)
    # calculate average movie rating score and rating count
    # 计算数值特征
    movieFeatures = ratingSamples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))
    print("movieFeatures的结构：")
    movieFeatures.printSchema()
    # movieFeatures.show(10)

    # bucketing
    # 分桶
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")

    # Normalization
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    print("movieProcessedFeatures 的结构：")
    movieProcessedFeatures.printSchema()
    # movieProcessedFeatures.show(10)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # file_path = 'file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources'
    # movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"esources'
    movieResourcesPath = r"E:\xn_work\xuyang\SparrowRecSys\target\classes\webroot\sampledata\movies.csv"
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    print("Raw Movie Samples--原始数据:")
    movieSamples.show(10)
    movieSamples.printSchema()
    print("OneHotEncoder Example--独热编码数据:")
    oneHotEncoderExample(movieSamples)
    print("MultiHotEncoder Example--多热编码数据:")
    multiHotEncoderExample(movieSamples)
    print("Numerical features Example--数值数据处理:")
    # ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"
    ratingsResourcesPath = r"E:\xn_work\xuyang\SparrowRecSys\target\classes\webroot\sampledata\ratings.csv"
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingFeatures(ratingSamples)
    """spark的milb库有点类似于sklearn库和pandas库的结合，满足了基本的模型对数据处理的要求"""
