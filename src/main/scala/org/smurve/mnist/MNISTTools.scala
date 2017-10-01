package org.smurve.mnist

import java.io.File
import java.util.UUID

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.mnist.config.MNistConfig
import org.smurve.nd4s._

import scala.util.Random

trait MNISTTools  {

  protected val config: MNistConfig
  protected val session: SparkSession
  protected val sc: SparkContext


  def saveModel(name: String, weights: Map[String, INDArray] ) : Unit = {

    println(s"Saving model as $name")
    for ( theta <- weights ) {
      Nd4j.saveBinary(theta._2, new File ( s"${name}_${theta._1}"))
    }
  }

  def readModel(name: String, weightNames: List[String]): Map[String, INDArray] = {
    val res = for (wn <- weightNames) yield {
      val weights = Nd4j.readBinary( new File(s"${name}_$wn"))
      wn->weights
    }
    res.toMap
  }

  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTImages wrappers
    */
  protected def createImagesFromBinary(name: String): RDD[MNISTImages] = {

    val rawImages = sc.binaryFiles(resolve(name))

    rawImages.map(p => {
      val stream = p._2
      new MNISTImages(stream.toArray())
    })
  }

  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTLabel wrappers
    */
  protected def createLabelsFromBinary(name: String): RDD[MNISTLabels] = {

    val rawLabels = sc.binaryFiles(resolve(name))

    rawLabels.map(p => {
      val stream = p._2
      new MNISTLabels(stream.toArray())
    })
  }

  /**
    * Read the data files
    * @param key either "train" or "test"
    * @return a pair of INDArrays containing images and labels
    */
  def readFromBinary(key: String): (INDArray, INDArray) = {
    val images: MNISTImages = createImagesFromBinary(s"input/$key").first()
    val labels = createLabelsFromBinary(s"input/$key-labels").first()

    (images.asINDarray, labels.asINDarray)
  }



  /**
    * resolve in local fs or hdfs, depending on config
    *
    * @param name short name of the file
    * @return
    */
  protected def resolve(name: String): String = config.prefix + name


  def asImageString(iNDArray: INDArray): String = {
    val arr = toArray(iNDArray)
    MNISTImage(arr.map(d => d.toByte), 28, 28).toString
  }

  protected def hdfs_save(rdd: RDD[MNISTImage], baseName: String): String = {
    val tmpName = baseName + "_" + UUID.randomUUID().toString
    rdd.saveAsObjectFile(tmpName)
    tmpName
  }

  /**
    *
    * @param numTrain number of training records/labels to read from file
    * @param numTest number of test records/labels to read from file
    * @param rnd a random generator
    * @return a pair of pairs containing training samples and labels and test samples and labels
    */
  def  readMNIST(numTrain: Int, numTest: Int, rnd: Random = new Random()
                ): ((INDArray, INDArray), Option[(INDArray, INDArray)]) = {

    println("Reading images from file...")
    val (img_test, lbl_test) = readFromBinary("/mnist/test")
    val trainingSet_orig = readFromBinary("/mnist/train")
    println("Done.")
    println("Shuffling...")
    val (img_train, lbl_train) = shuffle(trainingSet_orig, random = rnd)
    println("Done.")
    val trainingSet = (img_train(0 -> numTrain, ->), lbl_train(0 -> numTrain, ->))
    val testSet = if (numTest > 0 )
      Some((img_test(0 -> numTest, ->), lbl_test(0 -> numTest, ->)))
    else
      None

    (trainingSet, testSet)
  }

}
