from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import explode
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

MAX_MEMORY = "8g"

spark = SparkSession.builder.appName('recreation.gov reservations') \
    .config("spark.executor.memory", MAX_MEMORY) \
    .config("spark.driver.memory", MAX_MEMORY) \
    .getOrCreate()

schemaRating = StructType([
    StructField("item", IntegerType(), True),
    StructField("user", IntegerType(), True),
    StructField("rating", IntegerType(), True),
])

dfReservations2021 = spark.read.format('csv').schema(schemaRating).csv('REC_Collaborative_Facility.csv', header=True, ignoreTrailingWhiteSpace=True)

# split train and test
trainDF, testDF = dfReservations2021.randomSplit([0.8, 0.2])
trainDF.cache()

# build model
# coldStartStrategy - helped drop nulls
als = ALS(coldStartStrategy="drop", implicitPrefs=True)

# evalute model using root mean squared evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")

# set parameters for tuning
paramGrid = ParamGridBuilder()\
    .addGrid(als.maxIter, [5, 10, 25])\
    .addGrid(als.rank, [10, 50, 100])\
    .addGrid(als.regParam, [0.001, 0.01, 0.1])\
    .build()

crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator)

# cross validate create best model
cvModel = crossval.fit(trainDF)

# display best model
print(cvModel.bestModel)

# assess prediction model
cvPred = cvModel.bestModel.transform(testDF)
print(evaluator.evaluate(cvPred))

spark.stop()