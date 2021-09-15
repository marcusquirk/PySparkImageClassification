from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.util import MLWritable
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MulticlassLogisticRegression").config('spark.executor.memory', '60g').config('spark.driver.memory', '59g').config('spark.memory.offHeap.enabled', True).config('spark.memory.offHeap.size', '58g').config('spark.driver.maxResultSize', '57g').getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load("out.txt")
    lr = LogisticRegression(maxIter=4, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for multinomial logistic regression
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))

    trainingSummary = lrModel.summary
    print(type(trainingSummary))
    print(trainingSummary)
    
    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # for multiclass, we can inspect metrics on a per-label basis
    print("Precision by label:")
    for i, prec in enumerate(trainingSummary.precisionByLabel):
        print("label %d: %s" % (i, prec))

    print("Recall by label:")
    for i, rec in enumerate(trainingSummary.recallByLabel):
        print("label %d: %s" % (i, rec))

    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

    dustin = Pipeline(stages=[lrModel])
    katie = dustin.fit(training)
    katie.write().save("./model")

    spark.stop()
