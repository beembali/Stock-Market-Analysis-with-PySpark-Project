# Stock Market Data Pipeline and Analysis
import sys
import requests
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, FloatType, IntegerType, MapType, DoubleType, LongType, ArrayType
from pyspark.sql.functions import when, lag, from_json, explode, col, lit, to_date, year, month, day, avg, stddev, datediff, to_timestamp, mean, max as spark_max, min as spark_min
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F


#-------------- STEP 1 - INGESTION ------------------------------
#---------------- SPARK SESSION ---------------------

spark = SparkSession.builder \
    .appName("Single Stock API Data Processing") \
    .getOrCreate()


def fetch_stock_data(stock_endpoint, stock_symbol):
    alpha_api_key = os.environ.get('ALPHA_API_KEY')
    alpha_parameters = {"function": "TIME_SERIES_DAILY",
                        "symbol": stock_symbol,
                        "outputsize": "compact",
                        "apikey": alpha_api_key}
    response = requests.get(url=stock_endpoint, params=alpha_parameters)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed with status code {response.status_code}")


stock_endpoint= "https://www.alphavantage.co/query"
stock_symbol = "SPY"
stock_data = fetch_stock_data(stock_endpoint, stock_symbol)

#------------------- STEP 2 TRANSFORMATION ----------------
# Assuming the API returns a JSON with a 'data' field containing an array of data points
# Adjust the schema according to your actual API response


# Schema for the daily price data
daily_price_schema = StructType([
    StructField("1. open", StringType(), True),
    StructField("2. high", StringType(), True),
    StructField("3. low", StringType(), True),
    StructField("4. close", StringType(), True),
    StructField("5. volume", StringType(), True)
])

# Schema for the Time Series (Daily) data
time_series_schema = MapType(StringType(), daily_price_schema)

# Schema for the Meta Data
meta_data_schema = StructType([
    StructField("1. Information", StringType(), True),
    StructField("2. Symbol", StringType(), True),
    StructField("3. Last Refreshed", StringType(), True),
    StructField("4. Output Size", StringType(), True),
    StructField("5. Time Zone", StringType(), True)
])


stock_data_schema = StructType([
        StructField("Meta Data", meta_data_schema, True),
        StructField("Time Series (Daily)", time_series_schema, True)
    ])

# Overall schema for the JSON array
overall_schema = StructType(stock_data_schema)

# Create a DataFrame from the API response
df = spark.read.schema(overall_schema).json(spark.sparkContext.parallelize([stock_data]))

# Explode the "Time Series (Daily)" map as the key is the date the value is the stock data.
exploded_df = df.select(explode(("Time Series (Daily)")).alias("date", "stock_data"))


# Select and rename the columns, casting to appropriate types
final_df = exploded_df.select(
    to_date(exploded_df["date"]).alias("date"),
    exploded_df.stock_data["4. close"].alias("close"),
    exploded_df.stock_data["5. volume"].alias("volume")
).withColumn("year", year("date")) \
                   .withColumn("month", month("date")) \
                   .withColumn("day", day("date"))


#------------------------ STEP 3 ANALYSIS -------------------------------

#----------------------- Forecasting with Pyspark -------------------

# time series analysis to calculate moving averages, Bollinger Bands, or other technical indicators.
# Convert 'close' and 'volume' to numeric type
final_df = final_df.withColumn("close", col("close").cast(DoubleType()))
final_df = final_df.withColumn("volume", col("volume").cast(DoubleType()))

# Short-term moving average - 20 Days
windowSpec_20 = Window.orderBy(col("date")).rowsBetween(-19,0)
final_df = final_df.withColumn("moving_avg_20", avg(col("close")).over(windowSpec_20))

# Long-term moving average - 100 Days
windowSpec_100 = Window.orderBy(col("date")).rowsBetween(-99, 0)
final_df = final_df.withColumn("moving_avg_100", avg(col("close")).over(windowSpec_100))

# Bollinger Bands
final_df = final_df.withColumn("stddev_20", stddev(col("close")).over(windowSpec_20))

# Define upper and lower bollinger bands
final_df = final_df.withColumn("Upper_BB", col("moving_avg_20") + 2 * col("stddev_20"))\
    .withColumn("Lower_BB", col("moving_avg_20") - 2 * col("stddev_20"))

# Add previous close and volume (i.e. lag features)
window = Window.orderBy(col("date"))
final_df = final_df.withColumn("prev_close", lag("close", 1).over(window))
final_df = final_df.withColumn("prev_volume", lag("volume", 1).over(window))

# ROC (momentum)
final_df = final_df.withColumn("momentum_14", (col("close") - lag("close", 14).over(window)) / lag("close", 14).over(window))

# RSI (14-day)
final_df = final_df.withColumn("price_change", col("close") - col("prev_close"))
final_df = final_df.withColumn("gain", when(col("price_change") > 0, col("price_change")).otherwise(0))
final_df = final_df.withColumn("loss", when(col("price_change") < 0, -col("price_change")).otherwise(0))
final_df = final_df.withColumn("avg_gain", avg("gain").over(window.rowsBetween(-13, 0)))
final_df = final_df.withColumn("avg_loss", avg("loss").over(window.rowsBetween(-13, 0)))
final_df = final_df.withColumn("rs", col("avg_gain") / col("avg_loss"))
final_df = final_df.withColumn("rsi", 100 - (100 / (1 + col("rs"))))

# Convert date to days since epoch
final_df = final_df.withColumn("days_since_epoch", datediff(final_df.date, to_timestamp(lit("1970-01-01")))).orderBy("days_since_epoch")

# Parameters for the model
params_cols = ["prev_close", "prev_volume", "moving_avg_20", "moving_avg_100", "Upper_BB", "Lower_BB", "momentum_14", "rsi", "days_since_epoch"]

# Values in each column that are Null should be replaced with the mean
for col_name in params_cols:
    mean_value = final_df.select(mean(col_name)).collect()[0][0]
    final_df = final_df.na.fill(mean_value, subset=[col_name])

print("Final DF with mean values as null")
final_df.sort('date', ascending=[True]).show(5)

#---------------------- Linear Regression -----------------------------

assembler = VectorAssembler(inputCols=params_cols, outputCol="features")
df = assembler.transform(final_df)
final_data = df.select("date", "features", "close")
print("Final DF after selecting only the features and close column")
final_data.show(5)

#---- Time-based spliting --------

# Calculate the total number of days in the dataset
min_date = final_data.select(spark_min(col("date"))).collect()[0][0]
max_date = final_data.select(spark_max(col("date"))).collect()[0][0]

# Calculate total days using a temporary DataFrame
temp_df = final_data.select(
    to_date(lit(max_date), "yyyy-MM-dd").alias("max_date"),
    to_date(lit(min_date), "yyyy-MM-dd").alias("min_date")
)

total_days = temp_df.select(datediff(col("max_date"), col("min_date"))).collect()[0][0]

# Calculate the cutoff date (80% for training, 20% for testing)
days_for_training = int(total_days * 0.8)
cutoff_date = min_date + F.expr(f"INTERVAL {days_for_training} DAYS")

# Split the data
trainingDF = final_data.filter(col("date") <= cutoff_date)
testDF = final_data.filter(col("date") > cutoff_date)

num_partitions = 10  # Adjust based on your cluster and data size
trainingDF = trainingDF.repartition(num_partitions).sortWithinPartitions(col("date"), ascending=False)
testDF = testDF.repartition(num_partitions).sortWithinPartitions(col("date"), ascending=False)

# Cache the DataFrame to improve performance of subsequent operations
trainingDF.cache()
testDF.cache()

# Create the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="close")

# Fit the model to the training data
model = lr.fit(trainingDF)

# Generate predictions using the linear regression model for all features in the test dataframe:
predictions = model.transform(testDF).cache()
predicitions_plot = predictions.select("date", "prediction")

print("DF with predictions")
predictions.show(5)

# Evaluate Model
evaluator = RegressionEvaluator(labelCol='close', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data: {:.3f}".format(rmse))
print("R-squared (R2) on test data: {:.3f}".format(r2))

# Models Coefficients and Intercept
coefficients = model.coefficients
intercept = model.intercept

print("Coefficients: ", coefficients)
print("Intercept: {:.3f}".format(intercept))


# Determine which parameters in the params_cols impact the predictions most
feature_importance = sorted(list(zip(df.columns[:], map(abs, coefficients))), key=lambda x: x[1], reverse=True)

print("Feature Importance:")
for feature, importance in feature_importance:
    print("  {}: {:.3f}".format(feature, importance))

# Uncache when done
trainingDF.unpersist()
testDF.unpersist()

#------------ STEP 4 - Visualise with Matplotlib -------------------------
print("Visualisation of stock price in progress...")
import matplotlib.pyplot as plt

def create_analysis_plots(df, predictions, date_col='date'):
    # Merge predictions with original data
    merged_df = df.join(predictions, on=date_col, how='inner')

    # Sort by date
    merged_df = merged_df.orderBy(date_col)

    # Collect data for plotting (this step brings data to driver, use with caution for large datasets)
    plot_data = merged_df.select(date_col, 'close', 'prediction', 'moving_avg_20', 'moving_avg_100', 'Upper_BB',
                                 'Lower_BB').collect()
    dates = [row[date_col] for row in plot_data]
    actual_prices = [row['close'] for row in plot_data]
    predicted_prices = [row['prediction'] for row in plot_data]
    ma20_values = [row['moving_avg_20'] for row in plot_data]
    ma100_values = [row['moving_avg_100'] for row in plot_data]
    upper_bb = [row['Upper_BB'] for row in plot_data]
    lower_bb = [row['Lower_BB'] for row in plot_data]

    # Create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 28))
    fig.suptitle('Stock Price Analysis', fontsize=16)

    # 1. Actual vs Predicted Prices
    axs[0].plot(dates, actual_prices, label='Actual Close Price', color='blue')
    axs[0].plot(dates, predicted_prices, label='Predicted Close Price', color='red')
    axs[0].set_title('Actual vs Predicted Stock Prices')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Prediction Error Over Time
    errors = [a - p for a, p in zip(actual_prices, predicted_prices)]
    axs[1].plot(dates, errors, color='green', label='Prediction Errors')
    axs[1].axhline(y=0, color='r', linestyle='--')
    axs[1].set_title('Prediction Error Over Time')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Error (Actual - Predicted)')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Scatter Plot: Actual vs Predicted
    axs[2].scatter(actual_prices, predicted_prices, alpha=0.5)
    min_val = min(min(actual_prices), min(predicted_prices))
    max_val = max(max(actual_prices), max(predicted_prices))
    axs[2].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axs[2].set_title('Actual vs Predicted Values Scatter Plot')
    axs[2].set_xlabel('Actual Values')
    axs[2].set_ylabel('Predicted Values')
    axs[2].legend()

    # 4. Closing Price with Moving Averages and Bollinger Bands
    axs[3].plot(dates, actual_prices, label='Close Price', color='blue')
    axs[3].plot(dates, ma20_values, label='20-day MA', color='orange')
    axs[3].plot(dates, ma100_values, label='100-day MA', color='red')
    axs[3].plot(dates, upper_bb, label='Upper Bollinger Band', color='green', linestyle='--')
    axs[3].plot(dates, lower_bb, label='Lower Bollinger Band', color='green', linestyle='--')
    axs[3].fill_between(dates, upper_bb, lower_bb, alpha=0.1, color='green')
    axs[3].set_title('Closing Price with Moving Averages and Bollinger Bands')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Price')
    axs[3].legend()
    axs[3].grid(True)


    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.8)  # Adjust the top of the subplots
    plt.show()

create_analysis_plots(final_df, predicitions_plot)


# -------------- STEP 5 - Store DF in MySQL ---------------------------

# Spark DataFrame to load
final_df = final_df.select(col("date"), col('close'), col('volume'),col('year'),col('month'),col('day'))

# Repartition the data for parallel writing to MySQL
num_partitions = 10  # Adjust based on your cluster and data size
final_df = final_df.repartition(num_partitions, "year", "month").sortWithinPartitions('date', ascending=False)

# Parameters to connect to MySQL
hostname = 'localhost'
port = os.environ.get('sql_port')
database_name = "stock"
username = os.environ.get('sql_username')
password = os.environ.get('sql_password')
url = f"jdbc:mysql://{hostname}:{port}/{database_name}"

properties = {
    "user": f"{username}",
    "password": f"{password}",
    "driver": "com.mysql.cj.jdbc.Driver"
}
# Table name
table_name = stock_symbol.lower()

# Function to load existing data or return None if it doesn't exist
def load_existing_data(spark):

    try:
        # Check if table exists using metadata
        # Query the information schema to check if the table exists
        query = f"(SELECT table_name FROM information_schema.tables WHERE table_schema = '{database_name}' AND table_name = '{table_name}') AS table_check"

        # Execute the query
        table_check_df = spark.read.jdbc(url, query, properties=properties)
        # If table exists retrieve the latest data
        if table_check_df.count() > 0:
            query = f"(SELECT MAX(date) AS max_date FROM {table_name}) AS latest_date"
            # Read the latest date from MySQL
            latest_date_df = spark.read.jdbc(url, query, properties=properties)
            latest_date = latest_date_df.collect()[0]["max_date"]
            return latest_date
        else:
            print("Table may exists but no latest data is found, proceeding with inputting data into the table")
            return None

    except Exception as e:
        error_message = str(e).lower()  # Convert to lowercase for easier checking

        # Check for various "table doesn't exist" error messages
        table_not_exists_messages = [
            "table or view not found",
            "does not exist"
        ]

        if any(msg in error_message for msg in table_not_exists_messages):
            print("Table does not exist but will be created and data loaded")
            return None
        else:
            print(f"Error occurred while checking table or getting latest date: {str(e)}")


# Load existing data (if any)
existing_data = load_existing_data(spark)

if existing_data is not None:
    # Get the latest date in the existing data
    print(f"Latest date in existing data: {existing_data}")

    # Filter the new data to only include dates after the latest existing date
    new_data = final_df.filter(col("date") > existing_data)

    # Combine existing and new data
    updated_data = new_data
else:
    print("No existing data found. Saving all data as new.")
    updated_data = final_df

# Write the updated_data back to MySQL table
updated_data.write.jdbc(
    url=url,
    table=stock_symbol,
    mode="append",
    properties=properties
)

print("Data saved successfully.")

spark.stop()