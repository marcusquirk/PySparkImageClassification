#Marcus Quirk
#Dustin Johnson

from pyspark.sql import SparkSession
from pyspark.ml.util import MLReader
from pyspark.ml import PipelineModel

#Initiate the Spark session
spark=SparkSession.builder.appName('readModel').getOrCreate()

#Load the trained model and test data
lrModel = PipelineModel.load('./model')
testData = spark.read.format("libsvm").load("test_data.txt")

#Run the model on the test data
predictions = lrModel.transform(testData).collect()

#Print the predicted class for each entry in the test data
preds = [pred['prediction'] for pred in predictions]
print(preds)
