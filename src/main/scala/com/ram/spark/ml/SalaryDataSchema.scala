package com.ram.spark.ml

import org.apache.spark.sql.SparkSession
import Utils._

/**
  * Created by madhu on 18/5/16.
  */
object SalaryDataSchema {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
          master("local")
          .appName("example")
          .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePath)

    salaryDF.printSchema()

  }

}
