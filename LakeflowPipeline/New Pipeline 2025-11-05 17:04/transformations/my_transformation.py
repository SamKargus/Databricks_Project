import dlt
from pyspark.sql import functions as F

catalog = 'bronze_dev'
schema = 'dlt_demo_files'
volume = 'demo_synthetic_data'
subdir = 'file_ingestion'
VOLUME_PATH = f'/Volumes/{catalog}/{schema}/{volume}/{subdir}'

cloud_files_opts_json = {
  "cloudFiles.inferColumnTypes": "true",
  "cloudFiles.includeExistingFiles": "true",
  "cloudFiles.schemaEvolutionMode": "addNewColumns"
}
cloud_files_opts_csv = {
    **cloud_files_opts_json,
    'header': 'true'
}

@dlt.table(
    name='bronze_raw',
    comment='Raw union of Json and csv from volume',
    table_properties={'quality':'bronze'}
)
def bronze_raw():
    json_df = (
        spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'json')
        .option('cloudFiles.inferColumnTypes', 'true')
        .option('cloudFiles.includeExistingFiles', 'true')
        .option('cloudFiles.schemaEvolutionMode', 'addNewColumns')
        .load(f'{VOLUME_PATH}/*.json')
    )
    csv_df = (
        spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'csv')
        .option('cloudFiles.inferColumnTypes', 'true')
        .option('cloudFiles.includeExistingFiles', 'true')
        .option('cloudFiles.schemaEvolutionMode', 'addNewColumns')
        .option('header', 'true')
        .load(f'{VOLUME_PATH}/*.csv')
    )
    return json_df.select('*').unionByName(csv_df.select('*'), allowMissingColumns=True)

  
@dlt.table(
    name='silver_clean',
    comment='Normalized email and typed timestamps for cleaning',
    table_properties={'quality':'silver'}
)
@dlt.expect_or_drop('email_not_null', 'email IS NOT NULL')
@dlt.expect('id_must_exist', 'id IS NOT NULL')
def silver_clean():
  return (dlt.read_stream('bronze_raw')
          .withColumn('email_normalized', F.lower(F.trim(F.col('email'))))
          .withColumn('event_ts', F.to_timestamp('timestamp'))
          .withColumn('load_date', F.to_date(F.col('event_ts')))
          # watermark required for streaming aggregations
          .withWatermark('event_ts', '1 day')
        )

@dlt.table(
    name='gold_daily_users',
    comment='Daily approx users from silver layer (streaming-safe)',
    table_properties={'quality':'gold'}
)
def gold_daily_users():
    return (
        dlt.read_stream('silver_clean')
        .groupBy(
            F.window('event_ts', '1 day').alias('w')
        )
        .agg(
            F.approx_count_distinct('email_normalized').alias('unique_users')
        )
        .select(
            F.to_date(F.col('w.start')).alias('load_date'),
            F.col('unique_users')
        )
    )