package com.ram.spark.ml

import org.apache.spark.sql.SparkSession
import Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer

/**
  * Created by madhu on 18/5/16.
  */
object SalaryLabelIndexing {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
          master("local")
          .appName("example")
          .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePathTrain)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Estimator
    val labelIndexer = new StringIndexer()
    //specify options
    labelIndexer.setInputCol("salary")
    labelIndexer.setOutputCol("salary_index")

    //Model
    val labelIndexerTransformer = labelIndexer.fit(salaryDF)
    println("labels are "+ labelIndexerTransformer.labels.toList)

    //transformed data
    val transformedDF = labelIndexerTransformer.transform(salaryDF)
    transformedDF.select("salary","salary_index").show()



  }

}
