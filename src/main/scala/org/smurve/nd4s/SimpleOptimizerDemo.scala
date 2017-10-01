package org.smurve.nd4s

import grizzled.slf4j.Logging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.transform.Affine

import scala.util.Random

/**
  * Created by wgiersche on 26/07/17.
  */
object SimpleOptimizerDemo extends Logging {

  type LabeledSet = (INDArray, INDArray)
  val seed = 128L
  val rnd = new Random(seed)

  trait Setup {
    val (x1c, x2c) = (3, 3)
    val N_train = 20000
    val N_test = 1000
    val N_EPOCHS = 10
    val reportEvery = 5
    val eta: Double = 3e-4
    val nbatches = 5
    val task = false // task or data parallel

    val trSet: LabeledSet = createLabeledSet(N_train, x1c, x2c)
    val testSet: LabeledSet = createLabeledSet(N_test, x1c, x2c)

    val scale = 1.0
    val theta1: INDArray = (Nd4j.rand(seed, 3, 10) - .5) / scale
    val theta2: INDArray = (Nd4j.rand(seed, 11, 4) - .5) / scale
  }

  def main(args: Array[String]): Unit = {
    new Setup {

      info("=============================================================================================")
      info("                             Simple SGD Optimizer Demo ")
      info("=============================================================================================")

      val nn: Layer = Dense(theta1) !! ReLU() !! Dense(theta2) !! Sigmoid() !! Euclidean()

      val optimizer = new SimpleSGD(()=>Affine.identity, random = new Random(seed))

      optimizer.train(
        model = nn, nBatches = nbatches, parallel = 12, equiv = equiv10,
        trainingSet = trSet, testSet = Some(testSet),
        n_epochs = N_EPOCHS, eta = eta, reportEveryAfterBatches = reportEvery)
    }
  }

  /**
    * provide a list of points around the given center and identify the quadrant as in
    *
    * 2 | 4
    * ---+---
    * 1 | 3
    *
    * @param x1c the center x1 coord of the test grid
    * @param x2c the center x2 coord of the test grid
    * @return a pair consisting of the sample matrix and the labels matrix
    */
  def createLabeledSet(n: Int, x1c: Double, x2c: Double): (INDArray, INDArray) = {

    val samples = Nd4j.create(n, 2)
    val labels = Nd4j.create(n, 4)
    (0 until n).foreach { i =>
      val x1 = rnd.nextDouble * 10 - 5 + x1c
      val x2 = rnd.nextDouble * 10 - 5 + x2c
      val q = (if (x1 > x1c) 3 else 1) + (if (x2 > x2c) 1 else 0)
      val y_bar = appf(vec(1, 2, 3, 4), d => if (d == q) 1 else 0)
      samples(i) = vec(x1, x2)
      labels(i) = y_bar
    }
    (samples, labels)
  }

  def matches(y: INDArray, y_bar: INDArray): Boolean = {
    val found = (0 until y.length).map(y(_))
    val given = (0 until y.length).map(y_bar(_))
    (found zip given).map(p => p._2 == 1.0 && p._1 > 0.5).reduce(_ || _)
  }



}
