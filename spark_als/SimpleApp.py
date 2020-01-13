# 提交job
# ../spark-2.4.4-bin-hadoop2.7/bin/spark-submit   --master local[4]   SimpleApp.py

from pyspark.sql import SparkSession

logFile = "../hadoop-2.9.2/README.txt"  # Should be some file on your system
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
logData = spark.read.text(logFile).cache()

numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()
