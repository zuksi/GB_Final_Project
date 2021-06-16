# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer

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

new_ads = spark \
    .readStream \
    .format("csv") \
    .schema(my_schema) \
    .options(path="fp_stream", header=True,maxFilesPerTrigger=1) \
    .load()


def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='%s seconds' % freq) \
        .options(truncate=True) \
        .start()

s = console_output(new_ads, 5)
s.stop()

cassandra_companies = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="fraudulent_companies", keyspace="zybova") \
    .load()

cassandra_companies.show()

cvModel = PipelineModel.load("rf_model")

def writer_logic(df, epoch_id):
    df.persist()
    print("---------I've got new batch--------")
    print("New_ads:")
    df.show()
    df = df.na.fill(value="", subset=["location", "department", "company_profile", "description", "requirements",
                                      "benefits",
                                      "employment_type", "required_experience", "required_education", "industry",
                                      "function"])
    df = df.withColumn('full_description', F.concat_ws(" ", F.col('company_profile'),
                                                                 F.col('description'), F.col('requirements'),
                                                                 F.col('benefits')))
    df = df.withColumn('full_description_wc', F.size(F.split(F.col('full_description'), ' ')))

    df = df.withColumn('short_description', F.when(F.col("full_description_wc") < 300, 1)
                                 .otherwise(0))

    df = df.withColumn('ultra_short_description', F.when(F.col("full_description_wc") < 100, 1)
                                 .otherwise(0))
    df = df.withColumn('full_len', F.length(F.regexp_replace('full_description', " ", "")))
    df = df.withColumn('upper_description', F.regexp_replace('full_description', "[^A-Z]", ""))
    df = df.withColumn('upper_wc', F.size(F.split(F.col('upper_description'), ' ')))
    df = df.withColumn('upper_len', F.length(F.regexp_replace('upper_description', " ", "")))
    df = df.withColumn('upper_ratio_letters', F.col('upper_len') / F.col('full_len'))
    df = df.withColumn('tags', F.regexp_replace('full_description', "[^<>]", ""))
    df = df.withColumn('tags_count', F.length('tags'))
    df = df.withColumn('tags_ratio', F.col('tags_count') / F.col('full_description_wc'))
    df = df.withColumn('full_description_lower', F.lower(F.col('full_description')))
    df = df.withColumn('spamwords', F.when(F.col("full_description_lower").rlike("six figure salary"), 1)
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
                                 .when(F.col("full_description_lower").rlike("df\s*\w*\s*entry"), 1)
                                 .when(F.col("full_description_lower").rlike("typist"), 1)
                                 .when(F.col("full_description_lower").rlike("click here"), 1)
                                 .when(F.col("full_description_lower").rlike("work from home"), 1)
                                 .when(F.col("full_description_lower").rlike("internet access"), 1)
                                 .when(F.col("full_description_lower").rlike("clerk"), 1)
                                 .otherwise(0))
    df = df.withColumn('nohtml_benefits', (~(df.benefits.contains("<ul>")) &
                                                     ~(df.requirements.contains("<ol>")) &
                                                     ~(df.requirements.contains("<li>"))).cast('string'))

    company_indexer = StringIndexer(inputCol="company_profile", outputCol="company_id")
    df = company_indexer.fit(df).transform(df)
    df = df['full_description', 'short_description', 'ultra_short_description', 'upper_ratio_letters', 'tags_ratio',
        'spamwords', 'upper_wc', 'has_company_logo', 'nohtml_benefits', 'company_id', 'fraudulent']
    cassandra_companies.persist()
    print("Here is what I've got from Cassandra:")
    cassandra_companies.show()
    cassandra_companies_df = cassandra_companies.select("company_id").distinct()
    cassandra_companies_list_rows = cassandra_companies_df.collect()
    cassandra_companies_list = map(lambda x: x.__getattr__("company_id"), cassandra_companies_list_rows)
    list_companies = map(lambda row: row.asDict(), cassandra_companies.collect())
    dict_companies = {company['company_id']: company['fake_ads'] for company in list_companies}
    companies_list_rows_df = df.select("company_id").distinct().collect()
    companies_list_df = map(lambda x: x.__getattr__("company_id"), companies_list_rows_df)
    for company in companies_list_df:
        if company in cassandra_companies_list and dict_companies[company] > 1:
            cassandra_new = cassandra_companies.select(cassandra_companies['company_id'],
                                                       cassandra_companies['fake_ads'] + 1) \
                .where(F.col('company_id') == company)
            cassandra_new = cassandra_new.select(F.col('company_id'), F.col('(fake_ads + 1)').alias('fake_ads'))
            cassandra_new.show()
            cassandra_new.write \
                .format("org.apache.spark.sql.cassandra") \
                .options(table="fraudulent_companies", keyspace="zybova") \
                .mode("append") \
                .save()
            print("I updated data in Cassandra. Continue...")
            df = df.where(F.col('company_id') != company)
    predict = cvModel.transform(df).select('id', 'company_id', 'prediction')
    print("Predictions:")
    predict.show()
    predict_fake = predict.where((F.col('prediction') == 1.0) & (F.col('company_id') != 1710))
    predict_fake.show()
    predict_short = predict_fake.select(F.col("company_id"), F.col("prediction").alias("fake_ads"))
    predict_short.show()
    cassandra_stream_union = predict_short.union(cassandra_companies)
    cassandra_stream_aggregation = cassandra_stream_union.groupBy("company_id").agg(F.sum("fake_ads").alias("fake_ads"))
    print("Aggregated data:")
    cassandra_stream_aggregation.show()
    cassandra_stream_aggregation.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="fraudulent_companies", keyspace="zybova") \
        .mode("append") \
        .save()
    print("I saved the aggregation in Cassandra. Continue...")
    cassandra_companies.unpersist()
    df.unpersist()


stream = new_ads \
    .writeStream \
    .trigger(processingTime='100 seconds') \
    .foreachBatch(writer_logic) \
    .option("checkpointLocation", "checkpoints/fake_ads_checkpoint")

s = stream.start()

s.stop()