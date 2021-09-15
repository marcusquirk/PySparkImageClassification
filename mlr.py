#Marcus Quirk
#Dustin Johnson

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.util import MLWritable
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors

if __name__ == "__main__":
    
    #Create a spark session
    #The configuration options are necessary to allow the program enough memory (required to combat Java heap space error)
    spark = SparkSession.builder.appName("MulticlassLogisticRegression").config('spark.executor.memory', '60g').config('spark.driver.memory', '59g').config('spark.memory.offHeap.enabled', True).config('spark.memory.offHeap.size', '58g').config('spark.driver.maxResultSize', '57g').getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load("out.txt") #out.txt is the output from svmMaker

    #We did not have an opportunity to tweak the learning parameters
    lr = LogisticRegression(maxIter=4, regParam=0.3, elasticNetParam=0.8)

    #Fit the model
    lrModel = lr.fit(training)
    
    #The next lines produce evaluative statistics and print them to the terminal
    trainingSummary = lrModel.summary

    #Precision and recall are some of the most important Machine Learning statistics
    #This produces precision and recall by label
    #Precision is the percentage of instances that were correctly identified
    print("Precision by label:")
    for label, prec in enumerate(trainingSummary.precisionByLabel):
        print("label %d: %s" % (label, prec))

    #Recall is the percentage of the selected instances that were correct
    print("Recall by label:")
    for label, rec in enumerate(trainingSummary.recallByLabel):
        print("label %d: %s" % (label, rec))

    #Print statistics for the whole model
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("FPR: %s\nTPR: %s\nPrecision: %s\nRecall: %s"
          % (falsePositiveRate, truePositiveRate, precision, recall))

    #Save the model for later use
    modelPipeline = Pipeline(stages=[lrModel]) #LR models can't be saved unless they're in a Pipeline
    fittedModel = dustin.fit(training)
    fittedModel.write().save("./model")

    #End the Spark session
    spark.stop()
