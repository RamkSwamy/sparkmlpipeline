package com.ram.spark.ml

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by madhu on 18/5/16.
  */
object Utils {

  val filePathTrain = "src/main/resources/adult.data"
  val filePathTest = "src/main/resources/adult.test"

  def loadSalaryCsvTrain(sparkSession:SparkSession, path:String):DataFrame = {
    cleanDataFrame(sparkSession.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(path).toDF())
  }

  def loadSalaryCsvTest(sparkSession:SparkSession, path:String):DataFrame = {
    cleanDataFrame(sparkSession.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(path).toDF())
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
    val relationshipStages = buildOneHotPipeLine("relationship")
    val sexStages = buildOneHotPipeLine("sex")

    Array.concat(workClassPipeLineStages,educationPipelineStages,martialSatusStages,
                 occupationPipelineStages,relationshipStages,sexStages)

  }

  def cleanDataFrame(df:DataFrame):DataFrame = {
    val listOfColumns = List("workclass","occupation","native_country")
    val pattern ="?"

    val cleanedDF = listOfColumns.foldRight(df)((columnName,df) => {
      df.filter(s"trim(${columnName})" +  " <> '" + pattern +"'")
    })
    cleanedDF
  }

  def buildDataPrepPipeLine(sparkSession: SparkSession):Array[PipelineStage] = {

    val pipelineStagesforFeatures = Utils.buildPipeLineForFeaturePreparation(sparkSession)

    //Here we are using some numerical variables(ordinal) directly
    val assembler = new VectorAssembler()
      .setInputCols(Array("workclass_onehotindex", "occupation_onehotindex", "relationship_onehotindex",
        "marital_status_onehotindex","sex_onehotindex","age","education_num","hours_per_week"))
      .setOutputCol("features")

    val labelIndexer = new StringIndexer()
    //specify options
    labelIndexer.setInputCol("salary")
    labelIndexer.setOutputCol("label")

    val pipelineStagesWithAssembler = pipelineStagesforFeatures.toList ::: List(assembler,labelIndexer)

    pipelineStagesWithAssembler.toArray

  }



}
