package org.smurve.dl.experiments.stable

import java.io.File

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.dl.input.MNISTFileDataFactory
import org.smurve.nd4s._
import org.smurve.transform.Grid

import scala.util.Random

object NaiveMNISTDenseNetDemo {

  val NUM_TRAINING = 60000
  val NUM_TEST = 10000
  val SEED = 12345
  val HIDDEN_LAYER_SIZE = 200
  val IMAGE_SIZE: Int = 28 * 28
  val ETA = 1e-3
  val NUM_EPOCHS = 7
  var STORE_AS = "output/MNIST_DEMO"


  /**
    * @param args no args
    */
  def main(args: Array[String]): Unit = {

    println()
    println("=============================================================================================")
    println("                Solving MNIST with a dense network from bricks and mortar ")
    println("=============================================================================================")

    /**
      * The dense layers' parameter matrices contain the bias, also. That's what the + 1 is about
      */
    val theta1 = (Nd4j.rand(Array(IMAGE_SIZE + 1, HIDDEN_LAYER_SIZE), SEED) - 0.5) / 1000
    val theta2 = (Nd4j.rand(Array(HIDDEN_LAYER_SIZE + 1, 10), SEED) - 0.5) / 1000

    /**
      * create the network by stacking up some layers
      * Nice, ain't it?! All the plumbing and wiring for e.g. backprop is done behind the curtain
      */
    val denseNet = Dense(theta1) !! ReLU() !! Dense(theta2) !! Sigmoid() !! Euclidean()

    /**
      * read training dadta
      */
    val (trainingSet, testSet) = new MNISTFileDataFactory("input/mnist").rawData(NUM_TRAINING, NUM_TEST)

    val trainingSet_reshaped = (trainingSet._1.reshape(NUM_TRAINING, IMAGE_SIZE), trainingSet._2)
    val testSet_reshaped = (testSet._1.reshape(NUM_TEST, IMAGE_SIZE), testSet._2)

    /**
      * These parameters control monitoring output
      */
    denseNet.setParams(
      ("*:MaxPool", "print.output", 0),
      ("*:Conv", "print.output", 0),
      ("*:Conv", "print.stats", true),
      ("*:Dense", "print.stats", false)
    )

    /**
      * train the network mit Gradient Descent
      */
    new SimpleSGD().train(
      model = denseNet, nBatches = 60, parallel = 10, equiv = equiv10,
      trainingSet = trainingSet_reshaped, testSet = testSet_reshaped,
      n_epochs = NUM_EPOCHS, eta = ETA, reportEveryAfterBatches = 10)


    saveModel(STORE_AS, Map("Theta1" -> theta1, "Theta2" -> theta2))

    /**
      * do some work with it
      */
    readAndInferModel(STORE_AS, testSet)

  }

  /**
    * just for demo purpose: Read the weights and infer from the given samples, printing the first result
    *
    * @param name the base name of the parameter file
    */
  def readAndInferModel(name: String, test: (INDArray, INDArray)): Unit = {
    val (imgs, lbls) = test
    val rnd = new Random()

    val weights = readModel(name, List("Theta1", "Theta2"))
    val theta1_from_file = weights("Theta1")
    val theta2_from_file = weights("Theta2")
    val new_network: Layer = Dense(theta1_from_file) !! ReLU() !! Dense(theta2_from_file) !! Sigmoid() !! Euclidean()

    for (_ <- 0 to 10) {
      val idx = rnd.nextInt(imgs.size(0))
      val sample = imgs(idx, ->).reshape(1, IMAGE_SIZE)
      val res = new_network.ffwd(sample)
      println(new Grid(sample.reshape(28, 28)))
      val labeledAs = (lbls(idx, ->) ** vec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9).T).getInt(0)
      val classidAs = toArray(res).zipWithIndex.reduce((a, v) => if (v._1 > a._1) v else a)._2
      println(s"labeled as   : $labeledAs, classified as: $classidAs - $res")
    }
  }

  /**
    * Same the weight matrices as ND4J matrices
    *
    * @param name    the base name of the file
    * @param weights the weights to be saved
    */
  def saveModel(name: String, weights: Map[String, INDArray]): Unit = {

    println(s"Saving model as $name")
    for (theta <- weights) {
      Nd4j.saveBinary(theta._2, new File(s"${name}_${theta._1}"))
    }
  }

  /**
    * Read the saved matrices from the given file
    *
    * @param name        the base name of the files to be read
    * @param weightNames the names of the single weight matrices
    * @return
    */
  def readModel(name: String, weightNames: List[String]): Map[String, INDArray] = {
    println(s"Loading model $name from files")
    val res = for (wn <- weightNames) yield {
      val weights = Nd4j.readBinary(new File(s"${name}_$wn"))
      wn -> weights
    }
    res.toMap
  }


}
