# spark-submit random_forest_demo.py --input /Users/yanpengwang/Downloads/features_20170518 --output /Users/yanpengwang/Documents/FDT_GBRT/RF_Model
# load features, train a model and evaluate training error

import argparse

from keystone.features.trade_life import TradeLife
from keystone.features.trade_days import TradeDays
from keystone.features.trade_count import TradeCount
from keystone.features.pnl_efficiency import PnlEfficiency
from keystone.features.streak import Streak
from keystone.features.all_time_pnl import AllTimePnl
from keystone.features.avg_daily_pnl import AvgDailyPnl
from keystone.models.linear_regression_model import LinearRegressionModel

# from keystone.models.random_forest_model import RFRegressionModel

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def build_parser():
    parser = argparse.ArgumentParser(prog = 'demo_pipeline.py', description = 'pipeline which reads features and train a randomforest regression model')
    parser.add_argument('--master', dest = 'master', default = 'local[2]')
    parser.add_argument('--input', dest = 'input', required = True)
    parser.add_argument('--output', dest = 'output', required = True)
    parser.add_argument('--mode', dest = 'mode', default = 'ignore')
    return parser


if __name__ == '__main__':
    # load command line args and config
    cmd_args_parser = build_parser()
    args, _ = cmd_args_parser.parse_known_args()

    # load orders
    sc = SparkContext(master=args.master, appName='Demo Pipeline')
    spark = SparkSession(sc)
    reader = spark.read
    
    # load the feature matrix
    feature_matrix = reader.load(args.input)
    feature_clean = feature_matrix.dropna()
    
    feature_columns = ['trade_life', 'trade_days', 'trade_count',
                'pnl_efficiency', 'streak', 'all_time_pnl', 'avg_daily_pnl']
    
    va = VectorAssembler(inputCols = feature_columns, outputCol = 'features')
    trainingData = va.transform(feature_clean)
    
    rf = RandomForestRegressor(labelCol="potential_risk", featuresCol="features")
    potential_risk_model = rf.fit(trainingData)
        
    predictions_training = potential_risk_model.transform(trainingData)
    predictions_training.select("prediction", "potential_risk", "features").show(5)
        
    
    # Select (prediction, true label) and compute training error
    evaluator = RegressionEvaluator(
    labelCol="potential_risk", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions_training)
    print("Root Mean Squared Error (RMSE) on training data = %g" % rmse)


    # save model
    if (args.mode == 'overwrite'):
        potential_risk_model.write().overwrite().save(args.output)
    else:
        potential_risk_model.save(args.output)

    sc.stop()

