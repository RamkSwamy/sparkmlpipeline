package com.ram.spark.ml

import org.apache.spark.sql.SparkSession
import Utils._
import org.apache.log4j.{Level, Logger}


/**
  * Created by ram on 25/5/16.
  */
object SalaryDataExplore {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePathTrain)

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val resultSet = salaryDF.count()
    //total records or rows in the file => no of observations
    println("Total no of obsevations: " +resultSet)
    salaryDF.show(50)

    //describe
    salaryDF.describe("age","education_num","hours_per_week").show()
    //similar to confusion matrix
    salaryDF.stat.crosstab("sex","salary").show()

    val salaryGroup50K = salaryDF.groupBy("salary").count()
    salaryGroup50K.show()

  }

}
