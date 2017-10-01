package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * math-driven design of a network layer based on ND4J Linear Algebra implementation
  * A layer is designed as a parameterized function.
  * A network is a composition of layers, ending with an Outputlayer to the right and represented by the leftmost layer.
  * Forward pass and backprop must be implemented such that the subsequent layer is taken into account.
  * See e.g. Dense implementation for details
  * Note that obviously we only support linear models
  */
trait Layer {

  private var next: Option[Layer] = None
  private var prev: Option[Layer] = None


  /** automatically enumerate the layers in a network, starting with 1 */
  protected def seqno: Int = prev.map(_.seqno + 1).getOrElse(1)

  /**
    * present the next layer
    *
    * @return the next layer. There must be one, unless you override nextLayer in output layers
    */
  def nextLayer: Layer = next.getOrElse(throw new IllegalStateException(
    "Network shouldn't end here. Are you missing an output layer?"))


  /**
    * initialize with the knowledge of the subsequent layer
    *
    * @param subsequent the layer just attached to the right
    */
  def inititialize(subsequent: Layer): Unit = {
    next = Some(subsequent)
    subsequent.prev = Some(this)
  }

  /**
    * the function associated with this layer
    *
    * @param x the input vector
    * @return the function applied to the input vector
    */
  def fun(x: INDArray): INDArray


  /**
    * forward pass or inference: same for all layers
    *
    * @param x the batch of input row vectors
    * @return the associated output vector
    */
  def ffwd(x: INDArray): INDArray = {
    val y = fun(x)
    val value = nextLayer.ffwd(y)
    value
  }

  /**
    * transitively send params through the network
    *
    * @param params = a triple of ('layer name', 'param name', 'param value')
    *               the name is the simple class name plus the seqno
    */
  def setParams(params: (String, String, Any)*): Unit = {
    nextLayer.setParams(params: _*)
  }

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    */
  def fwbw(x: INDArray, y_bar: INDArray): BackPack

  /**
    * update the weights using the head of the Seq
    * implementers must forward the tail to the subsequent layers
    *
    * @param grads the amount to be added
    */
  def update(grads: Seq[INDArray]): Unit

  /**
    * Stacking operator with output.
    * Note that after composition, 'this' also represents the subsequent layer
    *
    * @param outputLayer an output layer
    * @return this
    */
  def !!(outputLayer: OutputLayer): Layer = {
    rightmost.inititialize(outputLayer)
    this
  }

  /**
    * @return the rightmost layer in the network or this, if there is none
    */
  def rightmost: Layer = next match {
    case Some(n) => n.rightmost
    case None => this
  }

  /**
    * Composing two layers
    *
    * @param next the subsequent layer
    * @return 'this', now representing the composition
    */
  def !!(next: Layer): Layer = {
    val r = rightmost
    r.inititialize(next)
    this
  }
}