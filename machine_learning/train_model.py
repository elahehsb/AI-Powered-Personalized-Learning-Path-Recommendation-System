from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName('StudentPerformancePrediction').getOrCreate()

# Load data from MongoDB
df = spark.read.format('mongo')\
    .option('uri', 'mongodb://localhost:27017/education_system.student_interactions')\
    .load()

# Feature extraction
assembler = VectorAssembler(inputCols=['student_id', 'content_id', 'score'], outputCol='features')
data = assembler.transform(df)

# Add a column for predicting learning style
data = data.withColumn('label', col('learning_style'))

# Train-test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Train a Random Forest model
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
rf_model = rf.fit(train_data)

# Evaluate the model
predictions = rf_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# Save the model
rf_model.save('models/student_learning_style_model')
