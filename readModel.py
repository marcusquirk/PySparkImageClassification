from pyspark.sql import SparkSession
from pyspark.ml.util import MLReader
from pyspark.ml import PipelineModel

spark=SparkSession.builder.appName('stupid').getOrCreate()
lrModel = PipelineModel.load('./model')
testData = spark.read.format("libsvm").load("sample_test_data.txt")
predictions = lrModel.transform(testData).collect()
preds = [x['prediction'] for x in predictions]
print(preds)
