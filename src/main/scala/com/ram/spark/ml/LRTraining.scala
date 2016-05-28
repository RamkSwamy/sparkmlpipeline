package com.ram.spark.ml

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import Utils._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression


/**
  * Created by ram on 25/5/16.
  */
object LRTraining {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePathTrain)

    //pipeline Estimator
    val pipelineStagesWithAssembler = buildDataPrepPipeLine(sparkSession)

    val pipeline = new Pipeline().setStages(pipelineStagesWithAssembler)
    //pipeline model for preparing data for LR
    val featurisedDF = pipeline.fit(salaryDF).transform(salaryDF)

    featurisedDF.select("features","label").show(truncate = false)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)

    //Logistic Regression Model
    val model = lr.fit(featurisedDF)

    println(model.intercept+ " "+model.coefficients)

    model.save("/tmp/lrmodelstore")

  }

}
