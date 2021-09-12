# necessary import
import io
import pandas as pd
#import tensorflow as tf
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from functools import reduce
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
#from tensorflow import keras
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
from IPython.display import display

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

# create a spark session
#from sparkdl import DeepImageFeaturizer

spark = SparkSession.builder.appName('DigitRecog').getOrCreate()

# loaded image
df = spark.read.format("image").load("./NumtaDB/training-a/", inferSchema = True)
print('hello')
images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("./NumtaDB/training-a/")

display(images.limit(5))
df.printSchema()

'''one = read.format("1").withColumn("label", lit(1))
two = read.format("2").withColumn("label", lit(2))
three = read.format("3").withColumn("label", lit(3))
four = read.format("4").withColumn("label", lit(4))
five = read.format("5").withColumn("label", lit(5))
six = read.format("6").withColumn("label", lit(6))
seven = read.format("7").withColumn("label", lit(7))
eight =read.format("8").withColumn("label", lit(8))
nine = read.format("9").withColumn("label", lit(9))

dataframes = [zero, one, two, three,four,
             five, six, seven, eight, nine]'''

'''# merge data frame
df = reduce(lambda first, second: first.union(second), dataframes)'''

# repartition dataframe 
df = df.repartition(200)

# split the data-frame
train, test = df.randomSplit([0.8, 0.2], 42)

# model: InceptionV3
# extracting feature from images
featurizer = sparkdl.DeepImageFeaturizer(inputCol="image",
                                 outputCol="features",
                                 modelName="InceptionV3")

# used as a multi class classifier
lr = LogisticRegression(maxIter=5, regParam=0.03, 
                        elasticNetParam=0.5, labelCol="label")

# define a pipeline model
sparkdn = Pipeline(stages=[featurizer, lr])
spark_model = sparkdn.fit(train) # start fitting or training



spark.stop()
