package com.ram.spark.ml

import org.apache.spark.sql.SparkSession
import Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap

/**
  * Created by ram on 25/5/16.
  */
object SalaryEvaluator {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val trainDF = loadSalaryCsvTrain(sparkSession,filePathTrain)

    val testDF = loadSalaryCsvTest(sparkSession,filePathTest)

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

   //Build pipeline Estimator
    val pipelineStagesWithAssembler = buildDataPrepPipeLine(sparkSession)

    val pipeline = new Pipeline().setStages(pipelineStagesWithAssembler)

    //Model generated and used to prepare data for LR
    val featurisedDFTrain = pipeline.fit(trainDF).transform(trainDF)

    val featurisedDFTest = pipeline.fit(testDF).transform(testDF)

    //Load the saved LR model
    val model = LogisticRegressionModel.load("/tmp/lrmodelstore")



    //predict using LR model using the prepared data thru the pipeline

    val testPredictions = model.transform(featurisedDFTest)
    val trainingPredictions = model.transform(featurisedDFTrain)

    //Let’s now evaluate our model using Area under ROC as a metric.

    val evaluator = new BinaryClassificationEvaluator()


    val evaluatorParamMap = ParamMap(evaluator.metricName -> "areaUnderROC")
    val aucTraining = evaluator.evaluate(trainingPredictions, evaluatorParamMap)
    println("AUC Training: " + aucTraining)

    val aucTest = evaluator.evaluate(testPredictions, evaluatorParamMap)
    println("AUC Test: " + aucTest)




  }

}
