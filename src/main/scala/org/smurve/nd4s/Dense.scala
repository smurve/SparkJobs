package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import scala.language.postfixOps

/**
  * Fully connected layer. Just needs to implement fwbw
  * @param theta the weight matrix including the bias as 0th row
  */
case class Dense(theta: INDArray) extends Layer with ParameterSupport with StatsSupport {

  def checkshape (x: INDArray): Unit = {
    require(x.rank == 2, "FCL Layer requires rank 2")
    require(x.size(1) == theta.size(0) - 1, s"row vectors should have size ${theta.size(0) - 1}")
  }

  def fun(x: INDArray): INDArray = {
    checkshape(x)
    h1(x) ** theta
  }

  def fwbw(x: INDArray, y_bar: INDArray): BackPack = {
    checkshape(x)
    val (dC_dy, grads, cost) = nextLayer.fwbw(fun(x), y_bar)

    val theta_t = theta(1->,->).T

    val dC_dx = dC_dy ** theta_t
    val grad = h1(x).T ** dC_dy
    (dC_dx, grad :: grads, cost)
  }

  /**
    * update from head and pass the tail on to subsequent layers
    * @param steps: The list of gradients accumulated during training
    */
  override def update(steps: Seq[INDArray]): Unit = {
    if (booleanParam("print.stats").getOrElse(false)) {
      printStats(s"Layer $seqno: {getClass.getSimpleName}", theta = theta, steps = steps.head)
    }
    theta += steps.head
    nextLayer.update(steps.tail)
  }
}

