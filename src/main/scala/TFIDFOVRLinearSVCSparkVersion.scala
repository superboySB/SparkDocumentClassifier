import com.hankcs.hanlp.dictionary.CustomDictionary
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.tokenizer.StandardTokenizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.MyOneVsRest.{OneVsRest, OneVsRestModel}
import org.apache.spark.ml.classification.Chart
import org.apache.spark.ml.feature.{HashingTF, IDF, IndexToString, StringIndexer, VectorIndexer}
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.{DateTime, Duration}

import scala.io.StdIn
import scala.collection.JavaConverters._

object TFIDFOVRLinearSVCSparkVersion {
  val train_data = "data/train.txt"
  val test_data = "data/test.txt"
  val numFeatures=2000
  val spark = SparkSession.builder().appName("Spark-one-vs-all-SVMwithSGD").master("local[8]").getOrCreate()
  val threshold_positive=1.0
  val threshold_negative=2.0

  def main(args: Array[String]): Unit = {

    // load a simple data file to try something.
//    val inputData = spark.read.format("libsvm").load(train_data)
//    val predictData = spark.read.format("libsvm").load(test_data)

    println("======Segment and Get Document Frequency======")
    val inputData=Preprocessing(url=train_data)
    val predictData=Preprocessing(url=test_data)

    // generate the train/test split.
    val Array(train, validation) = inputData.randomSplit(Array(0.8, 0.2))
    val test =predictData

    println("Need Parameters Adjust?(Y/N)")
    if (StdIn.readLine() == "Y") {
      println("=============Adjusting===================")
      parametersTunning(train, validation)
      System.exit(0)
    }

    println("==============Training===================")
    val (ovrModel,time)= trainModel(
      trainData = train,
      MaxIter = 15,
      RegParam = 0.01)
    val val_accuracy = evaluateModel(ovrModel, validation)
    println("One train finished, using time: " + time + " ms"+" Precision: "+val_accuracy)
    println()

    println("=============Validation===================")
    println(s"Train accuracy = $val_accuracy")

    println("============Test and Predict==============")
    // score the model on test data.
    val predictions = ovrModel.transform(test)
    // obtain evaluator.
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test accuracy = $accuracy")

    println("=======Find Different Species=============")
    val suspicion=predictions.where(s"Vote1<$threshold_positive and Vote4>(-1*$threshold_negative)")
      .select("filename","label","prediction","Vote1","Vote2","Vote3","Vote4")
    suspicion.show(numRows = 200,truncate = false)
    println("Num of Suspicion Document: "+suspicion.count())
    println("Search Rate: "+(suspicion.where("label=8.0").count()*1.0/(predictions.where("label=8.0").count()*1.0)).toDouble)
  }

  def Preprocessing(url: String): DataFrame = {
    import spark.implicits._
    // Segment, use Hanlp
    CustomDictionary.add("日  期")
    CustomDictionary.add("版  号")
    CustomDictionary.add("标  题")
    CustomDictionary.add("作  者")
    CustomDictionary.add("正  文")

    // Process data and segment
    val input_docs = spark.sparkContext.textFile(url).map { x =>
      val t = x.split(".txt\t")
      (t(0),(t(0).charAt(0).toInt - 48).toDouble, transform(t(1)))
    }.toDF("filename","label", "sentence_words")

    // Features extracting of data, using TF-IDF
    val hashingTF = new HashingTF().
      setInputCol("sentence_words").setOutputCol("rawFeatures").setNumFeatures(numFeatures)
    val featurizedData = hashingTF.transform(input_docs)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData).cache()

    rescaledData
  }

  def transform(sentence: String): List[String] = {
    val list = StandardTokenizer.segment(sentence)
    CoreStopWordDictionary.apply(list)
    val my_list = list.asScala
    my_list.map(x => x.word.replaceAll(" ", "")).toList
  }

  def trainModel(trainData: DataFrame, MaxIter: Int, RegParam: Double): (OneVsRestModel, Double) = {
    val startTime = new DateTime()
    // instantiate the base classifier
    val lsvc = new LinearSVC()
      .setFeaturesCol("features")
      .setMaxIter(MaxIter)
      .setRegParam(RegParam)
    // instantiate the One Vs Rest Classifier.
    val ovr = new OneVsRest()
      .setClassifier(lsvc)
      .setLabelCol("label")
    // train the multiclass model.
    val ovrModel = ovr.fit(trainData)
    println("A Train Process finished, Classify-Agents num: " + ovrModel.models.length)

    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)

    (ovrModel, duration.getMillis)
  }

  // Validation
  def evaluateModel(model: OneVsRestModel, validationData: DataFrame): Double = {
    // score the model on test data.
    val val_predictions = model.transform(validationData)
    // obtain evaluator.
    val val_evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    // compute the classification error on test data.
    val val_accuracy = val_evaluator.evaluate(val_predictions)

    val_accuracy
  }

  def evaluateParameter(trainData: DataFrame, validationData: DataFrame, evaluateParameter: String,
                        MaxIterArray: Array[Int], RegParamArray: Array[Double]) = {
    var dataBarChart = new DefaultCategoryDataset()
    var dataLineChart = new DefaultCategoryDataset()
    for (maxIter <- MaxIterArray;
         regParam <- RegParamArray) {
      val (model, time) = trainModel(trainData, maxIter, regParam)
      val auc = evaluateModel(model, validationData)
      val parameterData = evaluateParameter match {
        case "maxIter" => maxIter;
        case "regParam" => regParam
      }
      dataLineChart.addValue(time, "Time", parameterData.toString)
    }
    Chart.plotBarLineChart("SVM evaluations " + evaluateParameter, evaluateParameter, "AUC",
      0.58, 0.7, "Time", dataBarChart, dataLineChart)

  }

  def evaluateAllParameter(trainData: DataFrame, validationData: DataFrame, MaxIterArray: Array[Int],
                           RegParamArray: Array[Double]): Unit = {
    val evaluationsArray =
      for (maxIter <- MaxIterArray;
           regParam <- RegParamArray)
        yield {
          val (model, time) = trainModel(trainData, maxIter, regParam)
          val auc = evaluateModel(model, validationData)
          (maxIter, regParam, auc)
        }
    val BestEval = evaluationsArray.sortBy(_._3).reverse(0)
    println("Best Parameters--" + "maxIter: " + BestEval._1 + " ,regParam: " + BestEval._2 + " ,AUC: " + BestEval._3)

  }

  def parametersTunning(trainData: DataFrame, validationData: DataFrame): Unit = {
    evaluateParameter(trainData, validationData, "maxIter", Array(1, 5, 15, 25, 30, 40, 50), Array(0.01))
    evaluateParameter(trainData, validationData, "regParam", Array(25), Array(0.01, 0.1, 1))
    evaluateAllParameter(trainData, validationData,
      Array(1, 3, 5, 15, 25),
      Array(0.01, 0.1, 1))
  }

}

