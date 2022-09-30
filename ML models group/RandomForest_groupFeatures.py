import datetime
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
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

vector_features = ['nses', 'cltmoredata', 'clt1stappdata', 'dur_mean', 'dur_std', 'int_mean', 'int_std']
#'dur', 'nses', 'cltmoredata', 'clt1stappdata', 'dur_mean', 'dur_std', 'int_mean', 'int_std'

# Feature vectorization
vector_assembler = VectorAssembler(inputCols=vector_features, outputCol="features")
dataset = vector_assembler.transform(dataset)
vector_assembler_path = "../data/numeric_vector_assembler_RF_group.bin"
vector_assembler.write().overwrite().save(vector_assembler_path)
dataset.show(25)

# Create Decision Tree model, train and test
now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

rf = RandomForestClassifier(labelCol='label', featuresCol='features', seed=1234, impurity='entropy', maxDepth=10, numTrees= 0)
#  maxBins=32, maxDepth=15, numTrees= 30
rf = rf.fit(dataset)
model_output_path = "../data/RandomForest_group.bin"
rf.write().overwrite().save(model_output_path)

now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)


#Test the random forest model
test_dataset = spark.read.csv("../Dataset/group-dataset/dataset/test-dataset.csv", mode='DROPMALFORMED', schema=schema, header='true')
test_dataset = test_dataset.na.drop()
test_dataset.show()
finDataset = vector_assembler.transform(test_dataset)
finDataset.show()
result = rf.transform(finDataset)
result.show()

#See predictions in console
only_predictions = result.select("label", "prediction")
only_predictions.show()

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
prediction_df = only_predictions.toPandas()
prediction_df = prediction_df.replace([0, 1],["Legitimate", "Cobalt Strike Beacon"])
confusion_matrix = pd.crosstab(prediction_df["label"], prediction_df["prediction"],
                                   rownames=["Actual values"], colnames=["Prediction value"])
sn.heatmap(confusion_matrix, annot=True)
plt.show()