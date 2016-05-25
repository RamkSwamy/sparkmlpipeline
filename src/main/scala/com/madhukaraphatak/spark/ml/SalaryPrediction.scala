package com.madhukaraphatak.spark.ml
import Utils._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/**
  * Created by ram on 22/5/16.
  */
object SalaryPrediction {

 def main(args: Array[String]) {

  val sparkSession = SparkSession.builder.
    master("local")
    .appName("example")
    .getOrCreate()

  val salaryDF = loadSalaryCsv(sparkSession, filePath)
  val salaryTestDF= loadSalaryCsv(sparkSession, filePathTest)


  val rootLogger = Logger.getRootLogger
  rootLogger.setLevel(Level.ERROR)

  exploreData(salaryDF)
  //education fields already set
  salaryDF.groupBy("education","education_num").count().sort("education_num").show()

  val cleanSalaryDF = cleanDataFrame(salaryDF,"?","workclass")
  val cleanSalaryDF1 = cleanDataFrame(cleanSalaryDF,"?","occupation")
  val cleanSalaryDF2 = cleanDataFrame(cleanSalaryDF1,"?","native_country")

  val cleanSalaryTestDF = cleanDataFrame(salaryTestDF,"?","workclass")
  val cleanSalaryTestDF1 = cleanDataFrame(cleanSalaryTestDF,"?","occupation")
  val cleanSalaryTestDF2 = cleanDataFrame(cleanSalaryTestDF1,"?","native_country")

  val cleanedRecsTest = cleanSalaryTestDF2.count()

  println(cleanedRecsTest)



  //to find the min,max , mean etc
  cleanSalaryDF2.describe("age","education_num","hours_per_week").show()

  //similar to table command to find confusion matrix
  cleanSalaryDF2.stat.crosstab("sex","salary").show()

  //Salary groups
  val salaryGroup50K = salaryDF.groupBy("salary").count()
  salaryGroup50K.show()



  //setting the label

  val labelIndexer = new StringIndexer()
  //specify options
  labelIndexer.setInputCol("salary")
  labelIndexer.setOutputCol("label")

  val labelIndexerTransformer = labelIndexer.fit(cleanSalaryDF2)
  println("labels are "+ labelIndexerTransformer.labels.toList)

  // show transformed dataframe
  val transformedDF = labelIndexerTransformer.transform(cleanSalaryDF2)
  transformedDF.select("salary","label").show()
  transformedDF.show()

  val assembler = new VectorAssembler()
    .setInputCols(Array("workclass_onehotindex", "occupation_onehotindex", "relationship_onehotindex",
     "marital_status_onehotindex","sex_onehotindex","age","education_num","hours_per_week"))
    .setOutputCol("features")


  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.01)

  // val output = assembler.transform(cleanSalaryDF2)
  //println(output.select("features", "label").first())

  val labelingStage = labelIndexer.asInstanceOf[PipelineStage]
  val indexingAndEncodingStages = buildPipeLineForFeaturePreparation(sparkSession)
  val vectorAssemblerStage = assembler.asInstanceOf[PipelineStage]
  val logRegStage = lr.asInstanceOf[PipelineStage]


  val pipeline = new Pipeline()
    .setStages( Array.concat(Array(labelingStage),indexingAndEncodingStages,Array(vectorAssemblerStage,logRegStage)))

  val pipeLineModel = pipeline.fit(cleanSalaryDF2)

  val testPredictions = pipeLineModel.transform(cleanSalaryTestDF2)
  val trainingPredictions = pipeLineModel.transform(cleanSalaryDF2)

  println("trainingPrediction --------------------------")
  trainingPredictions.show()

  println("testPrediction --------------------------")
  testPredictions.show()

  testPredictions.select("prediction","label").show()


  cleanSalaryDF2.persist()
  cleanSalaryTestDF2.persist()

  // Print the coefficients and intercept for logistic regression
  // println(s"Coefficients: ${lrStage1.coefficients} Intercept: ${lrStage1.intercept}")


  val evaluator = new BinaryClassificationEvaluator()
  //Let’s now evaluate our model using Area under ROC as a metric.
  import org.apache.spark.ml.param.ParamMap
  val evaluatorParamMap = ParamMap(evaluator.metricName -> "areaUnderROC")
  val aucTraining = evaluator.evaluate(trainingPredictions, evaluatorParamMap)
  println("AUC Training: " + aucTraining)

  val aucTest = evaluator.evaluate(testPredictions, evaluatorParamMap)
  println("AUC Test: " + aucTest)

  //testPredictions.foreach(t => println(t(15)))

  //    val accura.cy = testPredictions.filter(testPredictions("label") === testPredictions("prediction")).count()
  //   println(accuracy)

  //correlation

  //val corEducation = testPredictions.stat.corr("education_num","education")

  //println("correlation between education fields" + corEducation)
  //Currently correlation calculation for columns with dataType StringType not supported.


  val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.01, 0.1, 1.0))
    .addGrid(lr.maxIter, Array(20, 30))
    .build()

  val crossValidator = new CrossValidator()
    .setEstimator(pipeline)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(5)
    .setEvaluator(evaluator)
  val crossValidatorModel = crossValidator.fit(cleanSalaryDF2)

  val newPredictions = crossValidatorModel.transform(cleanSalaryTestDF2)
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
