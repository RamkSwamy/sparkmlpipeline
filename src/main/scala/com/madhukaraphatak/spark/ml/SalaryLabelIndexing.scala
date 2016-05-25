package com.madhukaraphatak.spark.ml

import org.apache.spark.sql.SparkSession
import Utils._
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

    val salaryDF = loadSalaryCsv(sparkSession,filePath)

    val labelIndexer = new StringIndexer()
    //specify options
    labelIndexer.setInputCol("salary")
    labelIndexer.setOutputCol("salary_index")

    val labelIndexerTransformer = labelIndexer.fit(salaryDF)
    println("labels are "+ labelIndexerTransformer.labels.toList)

    // show transformed dataframe
    val transformedDF = labelIndexerTransformer.transform(salaryDF)
    transformedDF.select("salary","salary_index").show()



  }

}
