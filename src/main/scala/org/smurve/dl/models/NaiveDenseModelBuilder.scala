package org.smurve.dl.models

import java.io.File

import grizzled.slf4j.Logging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.nd4s.{Layer, _}

/**
  * creates a bare-bones self-made dense network
  */
object NaiveDenseModelBuilder extends Logging {

  /**
    * create a simple 3-Layer dense network
    *
    * @param seed       a seed for the random initialization of the weight matrices
    * @param imgSize    the width and height - must be the same
    * @param hiddenSize the size of the hidden layer
    * @return
    */
  def build(seed: Int,
            imgSize: Int = 28,
            hiddenSize: Int = 100
           ): Layer = {

    val inputSize = imgSize * imgSize

    val theta1 = Nd4j.rand(Array(inputSize + 1, hiddenSize), seed) - 0.5
    val theta2 = Nd4j.rand(Array(hiddenSize + 1, 10), seed) - 0.5

    // Nice, ain't it? All the plumbing and wiring for pass and back prop is cared for by the !!-operator
    Dense(theta1) !! ReLU() !! Dense(theta2) !! Sigmoid() !! Euclidean()
  }


}
