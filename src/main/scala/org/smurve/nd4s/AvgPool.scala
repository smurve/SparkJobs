package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
/**
  *
  * @param depth_stride the stride depth
  * @param height_stride the vertical stride size
  * @param width_stride the horizontal stride size
  */
case class AvgPool(depth_stride: Int, height_stride: Int, width_stride: Int) extends Layer with ParameterSupport {

  val N_values: Int = depth_stride * height_stride * width_stride

  /**
    * averaging over depth and stride, which represents the different feature map.
    * Note that we require the strides to perfectly fit into the input.
    *
    * @param x the input vector of rank 4: N_features x D x H x W
    * @return the function applied to the input vector
    */
  override def fun(x: INDArray): INDArray = {
    require(x.rank == 5, "Need to be rank 4: N_inp x N_features x D x H x W")
    require ( x.size(3) % height_stride == 0, "stride height doesn't divide input height.")
    require ( x.size(4) % width_stride == 0, "stride width doesn't divide input width.")

    val res = Nd4j.zeros(x.size(0), x.size(1), x.size(3) / height_stride, x.size(4) / width_stride)

    for {
      ni <- 0 until x.size(0)
      nf <- 0 until x.size(1)
      ir <- 0 until x.size(3) by height_stride
      ic <- 0 until x.size(4) by width_stride
    }
        res(ni, nf, ir/height_stride, ic/width_stride) = (for {
          d <- 0 until depth_stride
          r <- ir until ir + height_stride
          c <- ic until ic + width_stride
        } yield x(ni, nf, d,r,c)).sum / N_values

    printOutput(res)
    res
  }

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors, will be passed on to the output layer
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {
    val (dC_dy, grads, c) = nextLayer.fwbw(fun(x), y_bar)
    val dC_dx = Nd4j.zeros(x.shape: _*)

    for {
      n <- 0 until dC_dy.size(0)
      od <- 0 until dC_dy.size(1)
      or <- 0 until dC_dy.size(2)
      oc <- 0 until dC_dy.size(3)

      d <- 0 until depth_stride
      r <- or * height_stride until (or + 1) * height_stride
      c <- oc * width_stride until (oc + 1) * width_stride

      // chain rule again: 1/N = dy/dx
      } dC_dx(n, od, d, r, c) = dC_dy(n, od, or, oc) / N_values

    (dC_dx, grads, c)
  }


  /**
    * No params - nothing to do here. Just forward
    * @param grads the amount to be added
    */
  override def update(grads: Seq[INDArray]): Unit = nextLayer.update(grads)

  def numOutputVectors: Int = integerParam("print.output").getOrElse(0)


  def printOutput(array: INDArray): Unit = {
    val n = numOutputVectors
    if ( n > 0 ) {
      for (i <- 0 until n ) {
        val s = for {
          td <- 0 until array.size(1)
        } yield {
          visualize(array(i, td, ->, ->).reshape(array.size(2), array.size(3)))
        }
        println(in_a_row(" | ")(s: _*))
      }
    }
  }



}
