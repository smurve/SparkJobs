package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by wgiersche on 25/07/17.
  */
abstract class OutputLayer extends Layer {

  /**
    * The output layer is the only layer that does not have a next layer, obviously
    * @return
    */
  override def nextLayer: Layer = throw new NoSuchMethodException("Don't call nextLayer within an Output layer!")

  /**
    * ignore
    */
  override def setParams(params: (String, String, Any)*): Unit = {}

  /**
    * just pass the given value back
    * @param x the batch of input row vectors
    * @return the associated output vector
    */
  override def ffwd(x: INDArray): INDArray = x

  /**
    * pass-through function to comply to layer
    * @param y the outcome of the previous layer
    * @return just what's come in
    */
  override def fun(y: INDArray): INDArray = y

  /**
    * cost/loss function
    * @param y the output of the previous layer
    * @param y_bar the expected output
    * @return the cost at the given output y
    */
  def cost(y: INDArray, y_bar: INDArray ): Double

  /**
    * @param x the input vector
    * @param y_bar the expected outcome
    * @return the gradient of the cost function with respect to x and y_bar
    */
  def grad_c(x: INDArray, y_bar: INDArray): INDArray

  /**
    * @param rhs a layer
    * @return nothing
    * @throws UnsupportedOperationException, because this should be the last layer
    */
  override def !!(rhs: Layer ): Layer = {
    throw new UnsupportedOperationException("Output layer has no next layer.")
  }

  /**
    * start the back propagation by returning the cost function's gradient
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    * @return a PROPAGATED tuple
    */
  override def fwbw(x: INDArray, y_bar: INDArray): BackPack = (grad_c(x, y_bar), Nil, cost(x, y_bar))

  /**
    * update pass ends here. Check if grads is empty and do nothing, throw
    * @param grads: should be an empty list, since gradients should have been applied to previous layers already.
    */
  override def update(grads: Seq[INDArray]): Unit = {
    if ( grads.nonEmpty)
      throw new IllegalArgumentException("This is the output layer. There should be no more gradients to be applied.")
  }
}
