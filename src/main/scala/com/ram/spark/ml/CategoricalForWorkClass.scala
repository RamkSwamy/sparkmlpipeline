package com.ram.spark.ml

import Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * Created by madhu on 18/5/16.
  */
object CategoricalForWorkClass {

  def main(args: Array[String]) {

    //SparkSession is the new entry point to programming Spark with the Dataset and DataFrame API’s
    //It encompasses SQLContext, SparkContext etc.,

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val salaryDF = loadSalaryCsvTrain(sparkSession,filePathTrain)

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //pre-processing
    //Estimator
    val stringIndexer = new StringIndexer()
    //specify options
    stringIndexer.setInputCol("workclass")
    stringIndexer.setOutputCol("workclass_index")
    //Model
    val stringIndexerTransformer = stringIndexer.fit(salaryDF)
    println(s"labels for work class are ${stringIndexerTransformer.labels.toList} ")
    //transform
    val indexedDF = stringIndexerTransformer.transform(salaryDF)

    //run One hot encoding
    //Transformer
    val oneHotEncoder = new OneHotEncoder()
    oneHotEncoder.setInputCol("workclass_index")
    oneHotEncoder.setOutputCol("workclass_onehotindex")

    val oneHotEncodedDF = oneHotEncoder.transform(indexedDF)

    // show one hot encoding
    oneHotEncodedDF.select("workclass","workclass_index","workclass_onehotindex").show(truncate = false)
  }

}
