import datetime
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Initiate Spark session
conf= SparkConf().setAll([('spark.executor.memory', '4g'),('spark.driver.memory','4g')])
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

print("Viewing the beggining of the dataset...")

# Define schema of dataset
schema = StructType([
    StructField('src_ip', StringType(), True),
    StructField('src_p', IntegerType(), True),
    StructField('dst_ip', StringType(), True),
    StructField('dst_p', IntegerType(), True),
    StructField('proto',IntegerType() , True),
    StructField('dur', DoubleType(), True),
    StructField('nses', IntegerType(), True),
    StructField('cltmoredata', DoubleType(), True),
    StructField('clt1stappdata', DoubleType(), True),
    StructField('dur_mean', DoubleType(), True),
    StructField('dur_std', DoubleType(), True),
    StructField('int_mean', DoubleType(), True),
    StructField('int_std', DoubleType(), True),
    StructField('label', IntegerType(), True)])
# Load datasets
dataset = spark.read.csv("../Dataset/group-dataset/dataset/group-dataset.csv", mode='DROPMALFORMED', schema=schema, header='true')
dataset = dataset.na.drop()
dataset.show()


vector_features = ['dur', 'nses','cltmoredata', 'clt1stappdata', 'dur_mean', 'dur_std', 'int_mean','int_std']
#'dur','nses', 'cltmoredata', 'clt1stappdata','dur_mean', 'dur_std', 'int_mean', 'int_std'

# Feature vectorization
vector_assembler = VectorAssembler(inputCols= vector_features, outputCol="features")
dataset = vector_assembler.transform(dataset)
vector_assembler_path = "../data/numeric_vector_assembler_NN_group.bin"
#vector_assembler.write().overwrite().save(vector_assembler_path)
dataset.show(25)

# Create Neural network model
now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

#layers = [4, 12, 8, 2]
#layers = [5, 15, 10, 2]
#layers = [6, 18, 12, 2]
#layers = [7, 14, 14, 2]
layers = [8, 16, 16, 2]

nn = MultilayerPerceptronClassifier(layers=layers, labelCol='label', featuresCol='features', predictionCol='prediction',
                                    seed=7986706784745880246)
print(nn.getSeed())
# This seed detected 15 false positives for 8 nodes = 5744930274476935971
# This seed detected 89% TPR with 99.6% TNR = 7986706784745880246

nn = nn.fit(dataset)

model_output_path = "../data/NeuralNetwork_group.bin"
#nn.write().overwrite().save(model_output_path)


#Test the neural network model
test_dataset = spark.read.csv("../Dataset/group-dataset/dataset/test-dataset.csv", mode='DROPMALFORMED', schema=schema, header='true')
test_dataset = test_dataset.na.drop()
test_dataset.show()
finDataset = vector_assembler.transform(test_dataset)
finDataset.show()
result = nn.transform(finDataset)
result.show()

#See predictions in console
only_predictions = result.select("label", "prediction")

# Evaluate the results
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(result)
print("Accuracy = {}".format(accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
weightedPrecision = evaluator.evaluate(result)
print("weightedPrecision = {}".format(weightedPrecision))

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
f1 = evaluator.evaluate(result)
print("f1 = {}".format(f1))

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")
weightedRecall = evaluator.evaluate(result)
print("Recall = {}".format(weightedRecall))

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="truePositiveRateByLabel", metricLabel=0)
TPR0 = evaluator.evaluate(result)
print("TPR of Normal = {}".format(TPR0))

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="truePositiveRateByLabel", metricLabel=1)
TPR1 = evaluator.evaluate(result)
print("TPR of Attacks = {}".format(TPR1))

#Print a confusion matrix
p = result.filter(result.label == 1)
p.filter(p.prediction == 0).show()

q = result.filter(result.label == 0)
q.filter(q.prediction == 1).show()

prediction_df = only_predictions.toPandas()
prediction_df = prediction_df.replace([0, 1],["Legitimate", "Cobalt Strike Beacon"])
confusion_matrix = pd.crosstab(prediction_df["label"], prediction_df["prediction"],
                                   rownames=["Actual values"], colnames=["Prediction value"])
sn.heatmap(confusion_matrix, annot=True)
plt.show()


