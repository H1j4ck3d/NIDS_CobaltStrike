import datetime
from random import randint
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator,MulticlassClassificationEvaluator

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

vector_features = ['nses', 'cltmoredata', 'clt1stappdata']
#'dur', 'nses', 'cltmoredata', 'clt1stappdata', 'dur_mean', 'dur_std','int_mean', 'int_std'

# Feature vectorization
vector_assembler = VectorAssembler(inputCols= vector_features, outputCol="features")
dataset = vector_assembler.transform(dataset)
vector_assembler_path = "../data/numeric_vector_assembler_KM_group.bin"
vector_assembler.write().overwrite().save(vector_assembler_path)
dataset.show(25)


'''
#Feature scaling
scaler = StandardScaler(inputCol="features", outputCol="st_features")
scal = scaler.fit(dataset)
st_dataset = scal.transform(dataset)
scaler_path = "../data/scaler_KM_group.bin"
scal.write().overwrite().save(scaler_path)
st_dataset.show(25)
'''


# Create Kmeans model, train and test
now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

km = None
result = None

for i in range(10):
    seed = randint(0 , 100000000000)
    #dur seeds=91927527302, , 17662060349
    #no dur seeds=59857828079, 85420246877
    km = KMeans(k=2, featuresCol="features", seed=seed)
    km = km.fit(dataset)
    result = km.transform(dataset)

#Evaluate the results
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features", metricName="silhouette",
                                        distanceMeasure="squaredEuclidean")
    score = evaluator.evaluate(result)
    print("Silhouette score = {}".format(score))


now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

model_output_path = "../data/KMeans.bin"
km.write().overwrite().save(model_output_path)
print("The seed for this KMeans model is " + str(km.getSeed()))

#Test the k-means model
test_dataset = spark.read.csv("../Dataset/group-dataset/dataset/test-dataset.csv", mode='DROPMALFORMED', schema=schema, header='true')
test_dataset.show()

test_dataset = vector_assembler.transform(test_dataset)
test_dataset.show()
result = km.transform(test_dataset)
result.show()
result = result.withColumn("prediction", result.prediction.cast(DoubleType()))
result.groupBy("prediction").count().show()
prediction_df = result.select("label", "prediction").toPandas()

# Evaluate results
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
prediction_df = prediction_df.replace([0, 1],["Legitimate", "Cobalt Strike Beacon"])
confusion_matrix = pd.crosstab(prediction_df["label"], prediction_df["prediction"],
                                   rownames=["Actual values"], colnames=["Prediction value"])
sn.heatmap(confusion_matrix, annot=True)
plt.show()