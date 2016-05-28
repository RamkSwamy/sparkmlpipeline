package com.ram.spark.ml

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap

/**
  * Created by ram on 25/5/16.
  */
object SalaryCrossValidator {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local[*]")
      .appName("example")
      .getOrCreate()

    val trainDF = loadSalaryCsvTrain(sparkSession,filePathTrain)

    val testDF = loadSalaryCsvTest(sparkSession,filePathTest)

    trainDF.cache()

    testDF.cache()

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val lr = new LogisticRegression()

    val pipelineStagesWithAssembler = buildDataPrepPipeLine(sparkSession)

    //adding LogisticRegression estimator to the pipeline
    val pipelineWitLR = new Pipeline().setStages(pipelineStagesWithAssembler ++ Array(lr))


    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(lr.maxIter, Array(20, 30))
      .build()

    val evaluator = new BinaryClassificationEvaluator()

    //Cross Validation Estimator is constructed

    val crossValidator = new CrossValidator()
      .setEstimator(pipelineWitLR)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setEvaluator(evaluator)

    //Model is created
    val crossValidatorModel = crossValidator.fit(trainDF)
    //Model used to Predict
    val newPredictions = crossValidatorModel.transform(testDF)

    //Evaluate the Model
    val evaluatorParamMap = ParamMap(evaluator.metricName -> "areaUnderROC")

    val newAucTest = evaluator.evaluate(newPredictions, evaluatorParamMap)
    println("new AUC (with Cross Validation) " + newAucTest)
    val bestModel = crossValidatorModel.bestModel

    //Understand the Model selected
    println()
    println("Parameters for Best Model:")

    val bestPipelineModel = crossValidatorModel.bestModel.asInstanceOf[PipelineModel]
    val stages = bestPipelineModel.stages

    println("No of stages: " + stages.length)

    val lrStage = stages(14).asInstanceOf[LogisticRegressionModel]
    println("regParam = " + lrStage.getRegParam)



  }

}
