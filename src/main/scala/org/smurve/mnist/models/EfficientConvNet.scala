package org.smurve.mnist.models

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.nd4s._

class EfficientConvNet extends NeuralNet {

  /**
    * Starting already with meaningful weights in the conv layer
    */
  private val theta1 =  //Nd4j.rand(Array(8,4,3))
  //*
  vec (
    0,0,0,1,1,1,0,0,0,-1,-1,-1,
    0,0,0,-1,-1,-1,0,0,0,1,1,1,
    0,0,0,1,0,-1,1,0,-1,1,0,-1,
    0,0,0,-1,0,1,-1,0,1,-1,0,1,
    0,0,0,1,1,0,1,0,-1,0,-1,-1,
    0,0,0,-1,-1,0,-1,0,1,0,1,1,
    0,0,0,0,1,1,-1,0,1,-1,-1,0,
    0,0,0,0,-1,-1,1,0,-1,1,1,0
  ).reshape(8,4,3) / 6.0
   // */



  /** weights for the dense layers */
  private val theta3 = (Nd4j.rand(289, 200) - 0.5 ) / 400
  private val theta4 = (Nd4j.rand(201, 10) - 0.5) / 200

  val shape = Shape(1,1,28,28)
  val max1 = MaxPool(1, 2, 2)     // -> 14 x 14
  val conv1 = Conv(theta1, 1, 14, 14, 3) // -> 8 x 12 x 12
  val max2 = MaxPool(1, 2, 2) // 8 x 6 x 6
  val dense1 = Dense(theta3)
  val dense2 = Dense(theta4)

  override def model: Layer = shape !! max1 !! conv1 !! max2 !!
    Flatten(8, 6, 6) !! dense1 !! ReLU() !! dense2 !! Sigmoid() !! Euclidean()


}
