package com.ram.spark.ml

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import Utils._
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

    val trainDF = loadSalaryCsvTrain(sparkSession,filePath)

    val testDF = loadSalaryCsvTest(sparkSession,filePathTest)

    trainDF.cache()

    testDF.cache()

    val lr = new LogisticRegression()

    val pipelineStagesWithAssembler = buildDataPrepPipeLine(sparkSession)

    val pipelineWitLR = new Pipeline().setStages(pipelineStagesWithAssembler ++ Array(lr))


    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(lr.maxIter, Array(20, 30))
      .build()

    val evaluator = new BinaryClassificationEvaluator()


    val crossValidator = new CrossValidator()
      .setEstimator(pipelineWitLR)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setEvaluator(evaluator)
    val crossValidatorModel = crossValidator.fit(trainDF)

    val newPredictions = crossValidatorModel.transform(testDF)

    val evaluatorParamMap = ParamMap(evaluator.metricName -> "areaUnderROC")

    val newAucTest = evaluator.evaluate(newPredictions, evaluatorParamMap)
    println("new AUC (with Cross Validation) " + newAucTest)
    val bestModel = crossValidatorModel.bestModel

    println()
    println("Parameters for Best Model:")

    val bestPipelineModel = crossValidatorModel.bestModel.asInstanceOf[PipelineModel]
    val stages = bestPipelineModel.stages

    println("No of stages: " + stages.length)

    val lrStage = stages(14).asInstanceOf[LogisticRegressionModel]
    println("regParam = " + lrStage.getRegParam)



  }

}
