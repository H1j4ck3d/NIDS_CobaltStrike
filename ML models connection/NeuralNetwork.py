import datetime
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, StringIndexerModel
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
    StructField('id.orig_h', StringType(), True),
    StructField('id.orig_p', IntegerType(), True),
    StructField('id.resp_h', StringType(), True),
    StructField('id.resp_p',IntegerType(), True),
    StructField('proto',StringType() , True),
    StructField('service', StringType(), True),
    StructField('duration', DoubleType(), True),
    StructField('orig_bytes', IntegerType(), True),
    StructField('resp_bytes', IntegerType(), True),
    StructField('conn_state', StringType(), True),
    StructField('history', StringType(), True),
    StructField('orig_pkts',IntegerType(), True),
    StructField('orig_ip_bytes', IntegerType(), True),
    StructField('resp_pkts',IntegerType(), True),
    StructField('resp_ip_bytes', IntegerType(), True),
    StructField('label', IntegerType(), True)])

# Load dataset
dataset = spark.read.csv("../Dataset/conn-dataset/dataset.csv", mode='DROPMALFORMED', schema=schema, header='true')
dataset.show(25)


# Eliminate unnecesary features
features = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes',
            'resp_bytes', 'conn_state', 'history', 'orig_pkts', 'orig_ip_bytes','resp_pkts', 'resp_ip_bytes', 'label']

# Feature preprocessing
string_features = ['proto', 'service', 'history']

# Feature indexing
for feature in string_features:
    str_indexer = StringIndexer(inputCol=feature, outputCol=feature+'_index', handleInvalid='keep')
    dataset_model = str_indexer.fit(dataset)
    dataset = dataset_model.transform(dataset)
    str_indexer_model_path = "../data/str_indexer/str_indexer_model_NN_{}.bin".format(feature)
    dataset_model.write().overwrite().save(str_indexer_model_path)

dataset.show(25)
dataset.select("service","service_index").distinct().show()
dataset.select("history","history_index").distinct().show()

vector_features = ['proto_index', 'service_index', 'history_index', 'orig_bytes', 'resp_bytes', 'orig_pkts',
                   'orig_ip_bytes','resp_pkts', 'resp_ip_bytes']

# Feature vectorization
vector_assembler = VectorAssembler(inputCols= vector_features, outputCol="features")
dataset = vector_assembler.transform(dataset)
vector_assembler_path = "../data/numeric_vector_assembler_NN.bin"
vector_assembler.write().overwrite().save(vector_assembler_path)
dataset.show(25)

# Specify number of layers and nodes per layer
layers = [9, 18, 18, 2]

now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

# Create neural network model, train and test
mpc = MultilayerPerceptronClassifier(layers=layers, labelCol='label', seed=1234, featuresCol='features', predictionCol='prediction')
mpc = mpc.fit(dataset)
model_output_path = "../data/NeuralNetwork.bin"
#mpc.write().overwrite().save(model_output_path)

now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

#Test the neural network
test_dataset = spark.read.csv("../Dataset/conn-dataset/test-dataset.csv", mode='DROPMALFORMED', schema=schema, header='true')
test_dataset.show()

string_indexer_models = {}
for column in string_features:
        string_indexer_model_path = "../data/str_indexer/str_indexer_model_NN_{}.bin".format(column)
        string_indexer = StringIndexerModel.load(string_indexer_model_path)
        string_indexer_models[column] = string_indexer
for column in string_features:
    string_indexer_model = string_indexer_models[column]
    test_dataset = string_indexer_model.transform(test_dataset)

test_dataset = vector_assembler.transform(test_dataset)
test_dataset.show()
result = mpc.transform(test_dataset)
result.show()
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