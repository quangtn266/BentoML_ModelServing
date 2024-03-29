import sys
from awsglue.utils import getResolvedOptions
from awsglue.transforms import *
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import  StructType, StructField, StringType, IntegerType
from awsglue.dynamicframe import DynamicFrame

## @params: [JOB_NAME] add
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# spark context
sc = SparkContext()

# glue context
glueContext = GlueContext(sc)

# spark
spark = glueContext.spark_session

# job
job = Job(glueContext)

# initialize job
job.init(args['JOB_NAME'], args)

# glue context
read_gs = glueContext.create_dynamic_frame.from_catalog(
    database = "google_scholar",
    table_name="read"
)

print("Count: ", read_gs.count())

# print schema
read_gs.printSchema()

# convert to df
df_gs = read_gs.toDF()

# "author_name", "email", "affiliation", ..
df_gs.show(10)

# resaerch_interest can't be None
df_gs_clean = df_gs.filter("col4 != 'None'")

# referring Column Names
rdd_ri = df_gs_clean.rdd.map(lambda x: (x['col4']))

print("\nSample RDD rows:")
print(rdd_ri.take(5))
print("\n Sample RDD rows after freqenc count for reach words:")

# flatMap() helps to apply transformation
rdd_ri_freq = rdd_ri.flatMap(lambda x: [(w.lower(), 1) for w in x.split("##")]).reduceByKey(lambda a, b: a+b)

# print with take() function
print(rdd_ri_freq.take(5))

# convert to df with schema
schema = StructField([StructField("ri", StringType(), False),
                      StructField("frequency", IntegerType(), False)])

# convert rdd to df with schema
df = spark.createDataFrame(rdd_ri_freq, schema)
print("\n Proposed Schema of DF:")

# print schema (to verify)
df.printSchema()
print("\n RDD converted to DF with schema:")

# sort
df_sort = df.sort(F.col("frequency").desc())
df_sort.show(10, truncate=False)

# just one csv file
df_sort = df_sort.repartition(1)

# create a dynamic frame (equivalent to df) in glue context
dynamic_frame_write = DynamicFrame.fromDF(df_sort, glueContext, "dynamic_frame_write")

# path for output file
path_s3_write = "s3://google-scholar-csv/write/"

# write to s3
glueContext.write_dynamic_frame.from_options(
    frame = dynamic_frame_write,
    connection_type= "s3",
    connection_options= {
        "path": path_s3_write
    },
    format= "csv",
    format_options={
        "quoteChar": -1,
        "separator": "|"
    }
)

job.commit()