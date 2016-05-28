package com.ram.spark.ml
import Utils._
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession


/**
  * Created by madhu on 18/5/16.
  */
object SalaryVectorAssembler {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePathTrain)
    //Transformer
    val assembler = new VectorAssembler()
      .setInputCols(Array("workclass_onehotindex", "occupation_onehotindex", "relationship_onehotindex",
        "marital_status_onehotindex","sex_onehotindex","age","education_num","hours_per_week"))
      .setOutputCol("features")

    //Pipeline Estimator
    val pipelineStagesforFeatures = Utils.buildPipeLineForFeaturePreparation(sparkSession)

    val pipelineStagesWithAssembler = pipelineStagesforFeatures.toList ::: List(assembler)

    val pipeline = new Pipeline().setStages(pipelineStagesWithAssembler.toArray)
    //Model generated and data tronsformed
    val featurisedDF = pipeline.fit(salaryDF).transform(salaryDF)

    featurisedDF.select("features").show(truncate = false)

  }

}
