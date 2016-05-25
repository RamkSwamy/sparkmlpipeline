package com.madhukaraphatak.spark.ml

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by madhu on 18/5/16.
  */
object Utils {

  val filePath = "src/main/resources/adult.data"
  val filePathTest = "src/main/resources/adult.test"

  def loadSalaryCsv(sparkSession:SparkSession, path:String):DataFrame = {
    sparkSession.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(path).toDF()
  }

  private def buildOneHotPipeLine(colName:String):Array[PipelineStage] = {
    val stringIndexer = new StringIndexer()
      .setInputCol(s"$colName")
      .setOutputCol(s"${colName}_index")

    val oneHotEncoder = new OneHotEncoder()
      .setInputCol(s"${colName}_index")
      .setOutputCol(s"${colName}_onehotindex")

    Array(stringIndexer,oneHotEncoder)
  }

  def buildPipeLineForFeaturePreparation(sparkSession: SparkSession):Array[PipelineStage] = {
    //work class

    val workClassPipeLineStages = buildOneHotPipeLine("workclass")
    val educationPipelineStages = buildOneHotPipeLine("education")
    val occupationPipelineStages = buildOneHotPipeLine("occupation")
    val martialSatusStages = buildOneHotPipeLine("marital_status")
    val raltionshipStages = buildOneHotPipeLine("relationship")
    val sexStages = buildOneHotPipeLine("sex")

    Array.concat(workClassPipeLineStages,educationPipelineStages,martialSatusStages,
                 occupationPipelineStages,raltionshipStages,sexStages)

  }

  def cleanDataFrame(df:DataFrame,pattern:String, cols: String):DataFrame = {

    df.filter(s"trim(${cols})" +  " <> '" + pattern +"'")

  }

  def exploreData(df:DataFrame): Unit ={

    println("Data Schema" + df.toString())
    df.printSchema()
    val resultSet = df.count()
    //total records or rows in the file => no of observations
    println("Total no of obsevations" +resultSet)
    df.show(50)

  }

}
