# git clone --branch 3.1.0 https://github.com/JohnSnowLabs/spark-nlp

# export PYTHONPATH="./spark-nlp/python:$PYTHONPATH"
# export PYSPARK_PYTHON=/usr/bin/python3
# /spark2.4/bin/pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.0 --driver-memory 8g --driver-cores 4 --conf spark.memory.fraction="1" --master local[1]


from sparknlp.base import DocumentAssembler,Finisher
from sparknlp.annotator import Tokenizer,Normalizer,LemmatizerModel,StopWordsCleaner,Chunker,PerceptronModel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, StructField, IntegerType,DoubleType, ArrayType
from pyspark.ml.feature import IDF,CountVectorizer,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import VectorUDT

spark = SparkSession.builder \
   .appName("zybova_spark") \
   .master("local[1]") \
   .config("spark.driver.memory", "16G") \
   .config("spark.serializer",  "org.apache.spark.serializer.KryoSerializer") \
   .config("spark.kryoserializer.buffer.max", "2000M") \
   .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.3").getOrCreate()

spark = SparkSession.builder.appName("zybova_spark").getOrCreate()


new_schema = StructType([
    StructField(name='full_description', dataType=StringType(), nullable=False),
    StructField(name='short_description', dataType=IntegerType(), nullable=False),
    StructField(name='ultra_short_description', dataType=IntegerType(), nullable=False),
    StructField(name='upper_ratio_letters', dataType=DoubleType(), nullable=True),
    StructField(name='tags_ratio', dataType=DoubleType(), nullable=True),
    StructField(name='spamwords', dataType=IntegerType(), nullable=False),
    StructField(name='upper_wc', dataType=IntegerType(), nullable=False),
    StructField(name='has_company_logo', dataType=StringType(), nullable=True),
    StructField(name='nohtml_benefits', dataType=StringType(), nullable=False),
    StructField(name='company_id', dataType=IntegerType(), nullable=False),
    StructField(name='fraudulent', dataType=StringType(), nullable=True)])

test_set = spark \
    .read \
    .format("parquet") \
    .schema(new_schema) \
    .options(path="final_project/test_set", header=True, multiline=True) \
    .load()

test_set.show()

combined_train_set = spark \
    .read \
    .format("parquet") \
    .schema(new_schema) \
    .options(path="final_project/test_set", header=True, multiline=True) \
    .load()

combined_train_set.show()

stages=[]
for categoricalCol in ['has_company_logo','nohtml_benefits']:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    stages += [stringIndexer]

documentAssembler = DocumentAssembler().setInputCol("full_description").setOutputCol("document")

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("tokenized")

normalizer = Normalizer().setInputCols(['tokenized']).setOutputCol('normalized').setLowercase(True)

lemmatizer = LemmatizerModel.pretrained().setInputCols(['normalized']).setOutputCol('lemmatized')

stopwords_cleaner = StopWordsCleaner().setInputCols(['lemmatized']).setOutputCol('no_stop_lemmatized')

pos_tagger = PerceptronModel.pretrained('pos_anc').setInputCols(['document', 'lemmatized']).setOutputCol('pos')
allowed_tags = ['<JJ>+<NN>', '<NN>+<NN>','<VB>+<NN>','<NN>','<VB>+<NN>+<JJ>+<NN>','<VB>+<JJ>+<NN>']

chunker = Chunker().setInputCols(['document', 'pos']).setOutputCol('ngrams').setRegexParsers(allowed_tags)
finisher = Finisher().setInputCols(['ngrams'])

stages+=[documentAssembler,tokenizer,normalizer,lemmatizer,stopwords_cleaner,pos_tagger,chunker,finisher]

tfizer = CountVectorizer(inputCol='finished_ngrams',outputCol='tf_features')
idf = IDF(inputCol='tf_features', outputCol="features", minDocFreq=5)

stages+=[tfizer,idf]

assembler = VectorAssembler(inputCols = ['short_description','ultra_short_description','upper_ratio_letters',
 'tags_ratio','spamwords','upper_wc','has_company_logoIndex','nohtml_benefitsIndex','features'],
                            outputCol = "all_features")
scaler = StandardScaler(inputCol="all_features", outputCol="scaled_features")
labelIndexer = StringIndexer(inputCol="fraudulent", outputCol="label")

rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", numTrees=100)
stages+=[assembler,scaler,labelIndexer,rf]

pipeline = Pipeline(stages=stages)

paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [2, 5, 10]) \
    .addGrid(rf.maxBins, [5, 10, 20]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                          numFolds=5)

cvModel = crossval.fit(combined_train_set)
train_data_rf = cvModel.transform(combined_train_set)
test_data_rf = cvModel.transform(test_set)

AUC_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',labelCol='label',metricName='areaUnderROC')
AUC = AUC_evaluator.evaluate(test_data_rf)
print("The area under the curve is {}".format(AUC))

PR_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',labelCol='label',metricName='areaUnderPR')
PR = PR_evaluator.evaluate(test_data_rf)
print("The area under the PR curve is {}".format(PR))

tp = test_data_rf[(test_data_rf.label == 1) & (test_data_rf.prediction == 1)].count()
tn = test_data_rf[(test_data_rf.label == 0) & (test_data_rf.prediction == 0)].count()
fp = test_data_rf[(test_data_rf.label == 0) & (test_data_rf.prediction == 1)].count()
fn = test_data_rf[(test_data_rf.label == 1) & (test_data_rf.prediction == 0)].count()

print("tp: " + str(tp))
print("tn: " + str(tn))
print("fp: " + str(fp))
print("fn: " + str(fn))

cvModel.bestModel.write().overwrite().save("rf_model")