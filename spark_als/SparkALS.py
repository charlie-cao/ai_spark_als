# -*- coding: utf-8 -*-
# 官网地址: https://grouplens.org/datasets/movielens/ 
# ml-latest-small(1MB): http://files.grouplens.org/datasets/movielens/ml-latest-small.zip 
# ml-latest(234.2MB): http://files.grouplens.org/datasets/movielens/ml-latest.zip
# 注意需要把数据中的第一行删掉.
# 提交job
# 下载解压spark就好了,单机运行
# ../../spark-2.4.4-bin-hadoop2.7/bin/spark-submit --master local[4] SparkALS.py
# 模型参考 http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.recommendation
# Als 这几个参数的意义
# maxIter: max number of iterations 最大迭代次数
# regParam:regularization parameter 正则化参数
# coldStartStrategy = Param(parent='undefined', name='coldStartStrategy', doc="strategy for dealing with unknown or new users/items at prediction time. This may be useful in cross-validation or production scenarios, for handling user/item ids the model has not seen in the training data. Supported values: 'nan', 'drop'.")¶
# 在预测时间处理未知或新用户/项目的策略。这在交叉验证或生产场景中可能很有用，用于处理模型在培训数据中没有看到的用户/项目ID。支持的值：“nan”、“drop”。

# 协同过滤 (Collaborative Filtering, 简称 CF)，首先想一个简单的问题，如果你现在想看个电影，但你不知道具体看哪部，你会怎么做？大部分的人会问问周围的朋友，看看最近有什么好看的电影推荐，而我们一般更倾向于从口味比较类似的朋友那里得到推荐。这就是协同过滤的核心思想。
# 　　换句话说，就是借鉴和你相关人群的观点来进行推荐，很好理解。

# 如何保存模型.
# 如何使用模型,总不用每次都训练吧.
# 
# userId,movieId,rating,time
# 如果拿到google的数据能获得用户id和服务方id.以及评分,就可以获得
# 对某个用户的前十个推荐.基于协同过滤推荐算法,找出相似度最高的用户.
# 对某个服务方的前十个推荐.
# 不过这不是及时的.如果需要及时的话,需要每秒重新训练model.并使用新训练出来的model进行推荐.得用上流运算技术.
# 并且如果基于多个列的打分进行推荐.要扩展一下.
# userId,teamsId,rating,time
# 1,1,4.0,964982703
# 1,3,4.0,964981247
# 1,6,4.0,964982224
# 1,47,5.0,964983815
# 1,50,5.0,964982931
# 1,70,3.0,964982400
# 1,101,5.0,964980868
# 1,110,4.0,964982176
# 1,151,5.0,964984041
# 1,157,5.0,964984100

"""
Describe:     

"""

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.recommendation
from pyspark.sql import Row


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ALSExample").getOrCreate()

    # lines = spark.read.csv("/Users/caolei/Desktop/big-data/data/ml-latest-small/ratings.csv").rdd

    movies = spark.read.csv("/Users/caolei/Desktop/big-data/workspace/ai_spark_als/data/ml-latest-small/movies.csv").rdd.map(lambda l: Row(int(l[0]), str(l[1]), str(l[2]))).toDF(["movieId", "title", "genres"])
    ratings = spark.read.csv("/Users/caolei/Desktop/big-data/workspace/ai_spark_als/data/ml-latest-small/ratings.csv").rdd.map(lambda l: Row(int(l[0]), int(l[1]), float(l[2]))).toDF(["userId", "movieId", "rating"])


    # parts = lines.map(lambda row: row.value.split("::"))
    # ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=long(p[3])))
    # ratings = spark.createDataFrame(ratingsRDD)

    (training, test) = ratings.randomSplit([0.8, 0.2])

    # 冷启动策略使用"drop"，不对NaN进行评估
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
                coldStartStrategy="drop")
    # 训练模型
    model = als.fit(training)

    # 验证模型准确度.数值
    predictions = model.transform(test)
    # https://blog.csdn.net/u011707542/article/details/77838588
    # 
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # 开始评估
    rmse = evaluator.evaluate(predictions)
    # 评估值
    # 均方根误差亦称标准误差
    # 
    print("Root-mean-square error = " + str(rmse))

    # 对每个用户推荐top 10的movie
    userRecs = model.recommendForAllUsers(10)
    # 对每部电影推荐top 10的user
    movieRecs = model.recommendForAllItems(10)

    # 为指定的用户组推荐top 10的电影
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 5)
    # 为指定的电影组推荐top 10的用户
    movies = ratings.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 5)
    userRecs.show(5,0)
    movieRecs.show(5,0)
    userSubsetRecs.show()
    movieSubSetRecs.show()
    # 保存模型
    model.save("/Users/caolei/Desktop/big-data/workspace/ai_spark_als/model/spark-als-model")

    # 加载模型
    # sameModel = RandomForestClassificationModel.load("myModelPath")
    spark.stop()