package com.ram.spark.ml

import org.apache.spark.sql.SparkSession
import Utils._
import org.apache.log4j.{Level, Logger}

/**
  * Created by madhu on 18/5/16.
  */
object SalaryDataSchema {

  def main(args: Array[String]) {


    val sparkSession = SparkSession.builder.
          master("local")
          .appName("example")
          .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePathTrain)
    //Understand the structure(schema) of the data
    salaryDF.printSchema()

  }

}
