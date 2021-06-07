# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, StructField, IntegerType
from pyspark.ml.feature import IDF,CountVectorizer,VectorAssembler, RegexTokenizer,StopWordsRemover,NGram
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("zybova_spark").getOrCreate()

my_schema = StructType([
    StructField(name='title', dataType=StringType(), nullable=True),
    StructField(name='location', dataType=StringType(), nullable=True),
    StructField(name='department', dataType=StringType(), nullable=True),
    StructField(name='salary_range', dataType=StringType(), nullable=True),
    StructField(name='company_profile', dataType=StringType(), nullable=True),
    StructField(name='description', dataType=StringType(), nullable=True),
    StructField(name='requirements', dataType=StringType(), nullable=True),
    StructField(name='benefits', dataType=StringType(), nullable=True),
    StructField(name='telecommuting', dataType=StringType(), nullable=True),
    StructField(name='has_company_logo', dataType=StringType(), nullable=True),
    StructField(name='has_questions', dataType=StringType(), nullable=True),
    StructField(name='employment_type', dataType=StringType(), nullable=True),
    StructField(name='required_experience', dataType=StringType(), nullable=True),
    StructField(name='required_education', dataType=StringType(), nullable=True),
    StructField(name='industry', dataType=StringType(), nullable=True),
    StructField(name='function', dataType=StringType(), nullable=True),
    StructField(name='fraudulent', dataType=StringType(), nullable=True),
    StructField(name='in_balanced_dataset', dataType=StringType(), nullable=True)])

# read the dataset
my_data = spark \
    .read \
    .format("csv") \
    .schema(my_schema) \
    .options(path="final_project", header=True, multiline=True, escape="\"") \
    .load()

my_data.show()

my_data = my_data.na.fill(value="",subset=["location", "department","company_profile","description","requirements","benefits",
                                    "employment_type","required_experience","required_education","industry","function"])
my_data = my_data.withColumn('full_description', F.concat_ws(" ", F.col('company_profile'),
                       F.col('description'), F.col('requirements'),F.col('benefits')))
my_data=my_data.withColumn('full_description_wc', F.size(F.split(F.col('full_description'), ' ')))

my_data = my_data.withColumn('short_description',F.when(F.col("full_description_wc") < 300, 1)
                       .otherwise(0))

my_data = my_data.withColumn('ultra_short_description',F.when(F.col("full_description_wc") < 100, 1)
                       .otherwise(0))
my_data=my_data.withColumn('full_len',F.length(F.regexp_replace('full_description', " ", "")))
my_data = my_data.withColumn('upper_description',F.regexp_replace('full_description', "[^A-Z]", ""))
my_data=my_data.withColumn('upper_wc', F.size(F.split(F.col('upper_description'), ' ')))
my_data=my_data.withColumn('upper_len',F.length(F.regexp_replace('upper_description', " ", "")))
my_data=my_data.withColumn('upper_ratio_letters', F.col('upper_len')/ F.col('full_len') )
my_data = my_data.withColumn('tags',F.regexp_replace('full_description', "[^<>]", ""))
my_data = my_data.withColumn('tags_count',F.length('tags'))
my_data=my_data.withColumn('tags_ratio', F.col('tags_count')/ F.col('full_description_wc') )
my_data = my_data.withColumn('full_description_lower',F.lower(F.col('full_description')))
my_data = my_data.withColumn('spamwords',F.when(F.col("full_description_lower").rlike("six figure salary"), 1)
                       .when(F.col("full_description_lower").rlike("adventur"), 1)
                       .when(F.col("full_description_lower").rlike("get\s*\w*\s*cash"), 1)
                       .when(F.col("full_description_lower").rlike("fantastic income"), 1)
                       .when(F.col("full_description").rlike("RIGHT NOW"), 1)
                       .when(F.col("full_description_lower").rlike("your own boss"), 1)
                       .when(F.col("full_description_lower").rlike("urgently"), 1)
                       .when(F.col("full_description_lower").rlike("quick advancement"), 1)
                       .when(F.col("full_description_lower").rlike("high earnings"), 1)
                       .when(F.col("full_description_lower").rlike("earliest"), 1)
                       .when(F.col("full_description_lower").rlike("img"), 1)
                       .when(F.col("full_description_lower").rlike("bright future"), 1)
                       .when(F.col("full_description_lower").rlike("________________________________________"), 1)
                       .when(F.col("full_description_lower").rlike("no\s*\w*\s*recruiters"), 1)
                       .when(F.col("full_description_lower").rlike("my_data\s*\w*\s*entry"), 1)
                       .when(F.col("full_description_lower").rlike("typist"), 1)
                       .when(F.col("full_description_lower").rlike("click here"), 1)
                       .when(F.col("full_description_lower").rlike("work from home"), 1)
                       .when(F.col("full_description_lower").rlike("internet access"), 1)
                       .when(F.col("full_description_lower").rlike("clerk"), 1)
                       .otherwise(0))
my_data = my_data.withColumn('nohtml_benefits',(~(my_data.benefits.contains("<ul>"))&
                      ~(my_data.requirements.contains("<ol>"))&
                      ~(my_data.requirements.contains("<li>"))).cast('string'))
data_new = my_data['full_description','short_description','ultra_short_description','upper_ratio_letters','tags_ratio',
                    'spamwords','upper_wc','has_company_logo','nohtml_benefits','fraudulent']

data_new.columns

data_new.write.option("header",True).csv("final_project/new_data")

(train_set, test_set) = data_new.randomSplit([0.70, 0.30], seed = 42)

train_set.write.option("header",True).csv("final_project/train_set")
test_set.write.option("header",True).csv("final_project/test_set")

# major_df = train_set.filter(train_set.fraudulent == "f")
# minor_df = train_set.filter(train_set.fraudulent == "t")
# ratio = int(major_df.count()/minor_df.count())
# print("ratio: {}".format(ratio))
# a = range(ratio)
#
# # duplicate the minority rows
# oversampled_df = minor_df.withColumn("dummy", F.explode(F.array([F.lit(x) for x in a]))).drop('dummy')
#
# # combine both oversampled minority rows and previous majority rows
# combined_train_set = major_df.unionAll(oversampled_df)