package com.madhukaraphatak.spark.ml
import Utils._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * Created by madhu on 18/5/16.
  */
object CategoricalForWorkClass {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val salaryDF = loadSalaryCsv(sparkSession,filePath)

    val stringIndexer = new StringIndexer()
    //specify options
    stringIndexer.setInputCol("workclass")
    stringIndexer.setOutputCol("workclass_index")

    val stringIndexerTransformer = stringIndexer.fit(salaryDF)
    println(s"labels for work class are ${stringIndexerTransformer.labels.toList} ")

    //run One hot encoding

    val indexedDF = stringIndexerTransformer.transform(salaryDF)

    val oneHotEncoder = new OneHotEncoder()
    oneHotEncoder.setInputCol("workclass_index")
    oneHotEncoder.setOutputCol("workclass_onehotindex")

    val oneHotEncodedDF = oneHotEncoder.transform(indexedDF)

    // show one hot encoding
    oneHotEncodedDF.select("workclass","workclass_index","workclass_onehotindex").show()

  }

}
