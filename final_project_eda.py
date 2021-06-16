# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, StructField, IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

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

data = spark \
    .read \
    .format("csv") \
    .schema(my_schema) \
    .options(path="final_project", header=True, multiline=True, escape="\"") \
    .load()

data.printSchema()

data.show(2)

print((data.count(), len(data.columns)))

print(data.where(data['fraudulent'] == "t").count())
print(data.where(data['fraudulent'] == "f").count())

data = data.drop("in_balanced_dataset")

for col in data.columns:
    print(col + " unique values: " + str(data.select([col]).distinct().count()))

for col in data.columns:
    print(col + " with null values: "+ str(data.filter(data[col].isNull()).count()))

data = data.na.fill(value="",
                    subset=["location", "department", "company_profile", "description", "requirements", "benefits",
                            "employment_type", "required_experience", "required_education", "industry", "function"])

indexer = StringIndexer(inputCol="company_profile", outputCol="company_id")
data = indexer.fit(data).transform(data)

data = data.withColumn('full_description', F.concat_ws(" ", F.col('department'), F.col('company_profile'),
                                                       F.col('description'), F.col('requirements'), F.col('benefits'),
                                                       F.col('employment_type'),
                                                       F.col('required_experience'), F.col('required_education'),
                                                       F.col('industry'), F.col('function')))

new_data = data.drop("location", "department", "company_profile", "salary_range", "description", "requirements",
                     "benefits",
                     "employment_type", "required_experience", "required_education",
                     "industry", "function", "country", "state", "city")

tel_fraud = new_data.stat.crosstab("fraudulent","telecommuting")
tel_fraud = tel_fraud.withColumn("f_rp", F.round(F.col("f") /
                                                 (F.col("f") + F.col("t")), 2)).withColumn("t_rp",
                                                                                           F.round(F.col("t") /
                                                                                                   (F.col("f") + F.col(
                                                                                                       "t")), 2))
tel_fraud.show()

logo_fraud = new_data.stat.crosstab("fraudulent", "has_company_logo")
logo_fraud = logo_fraud.withColumn("f_rp", F.round(F.col("f") /
                                                   (F.col("f") + F.col("t")), 2)).withColumn("t_rp",
                                                                                             F.round(F.col("t") /
                                                                                                     (F.col(
                                                                                                         "f") + F.col(
                                                                                                         "t")), 2))
logo_fraud.show()

questions_fraud = new_data.stat.crosstab("fraudulent", "has_questions")
questions_fraud = questions_fraud.withColumn("f_rp", F.round(F.col("f") /
                                                             (F.col("f") + F.col("t")), 2)).withColumn("t_rp",
                                                                                                       F.round(
                                                                                                           F.col("t") /
                                                                                                           (F.col(
                                                                                                               "f") + F.col(
                                                                                                               "t")),
                                                                                                           2))
questions_fraud.show()

questions_logo = new_data.stat.crosstab("has_company_logo", "has_questions")
questions_logo = questions_logo.withColumn("f_rp", F.round(F.col("f") /
                                                           (F.col("f") + F.col("t")), 2)).withColumn("t_rp",
                                                                                                     F.round(
                                                                                                         F.col("t") /
                                                                                                         (F.col(
                                                                                                             "f") + F.col(
                                                                                                             "t")), 2))
questions_logo.show()

new_data[['description']].where(new_data.fraudulent=="t").show(truncate=False)

len_data = data.withColumn('title_len', F.length('title')) \
    .withColumn('company_profile_len', F.length('company_profile')) \
    .withColumn('description_len', F.length('description')) \
    .withColumn('requirements_len', F.length('requirements')) \
    .withColumn('benefits_len', F.length('benefits')) \
    .withColumn('full_description_len', F.length('full_description'))

len_data = len_data["fraudulent", "title_len", "company_profile_len",
                    "description_len", "requirements_len", "benefits_len", "full_description_len"]

len_data.show()



label_indexer = StringIndexer(inputCol="fraudulent", outputCol="label")
len_data = label_indexer.fit(len_data).transform(len_data)

assembler = VectorAssembler(
    inputCols=['title_len',"company_profile_len","description_len", "requirements_len",
               "benefits_len", "full_description_len","label"],
    outputCol="features")

assembled = assembler.transform(len_data)

spearman_corr = Correlation.corr(assembled, "features", method='spearman')

corr_list = spearman_corr.head()[0].toArray().tolist()
spearman_corr_df = spark.createDataFrame(corr_list)
spearman_corr_df.show(truncate=False)
