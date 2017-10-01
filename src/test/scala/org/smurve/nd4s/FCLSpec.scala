package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.nd4s.Implicits._

class FCLSpec extends FlatSpec with ShouldMatchers {

  val theta: INDArray = vec(1, 1, 1, -2, -1, 0, 2, 3, 4).reshape(3, 3)

  "h1" should "pad horizontically" in {
    val theta = vec(2, 5, 8, 2, 9, 16).reshape(2, 3)
    h1(theta) shouldEqual vec(1, 2, 5, 8, 1, 2, 9, 16).reshape(2, 4)
  }

  "v1" should "pad vertically" in {
    val theta = vec(-2, -1, 0, 2, 3, 4).reshape(2, 3)
    v1(theta) shouldEqual vec(1, 1, 1, -2, -1, 0, 2, 3, 4).reshape(3, 3)
  }

  "An FCL feed forward" should "simmply perform matrix multiplication on the batch" in {
    val x = vec(1f, 2, 3, 4).reshape(2, 2)
    val network = Dense(theta) !! Euclidean()
    val y = network.ffwd(x)
    y shouldEqual vec(3, 6, 9, 3, 10, 17).reshape(2, 3)
  }

  lazy val propd: BackPack = {
    val x = vec(1, 2, 3, 4).reshape(2, 2)
    val output = Euclidean()

    val y_bar = vec(2, 6, 5, 3, 4, 6).reshape(2, 3)

    val network = Dense(theta) !! output
    network.fwbw(x, y_bar)
  }

  "A Euclidean" should "compute the cost correctly" in {
    val y = vec(2, 5, 8, 2, 9, 16).reshape(2,3)
    val y_bar = vec(2, 6, 5, 3, 4, 6).reshape(2, 3)
    val output = Euclidean()
    output.cost(y, y_bar) shouldEqual 68
    output.grad_c(y, y_bar) shouldEqual vec(0, -1, 3, -1, 5, 10).reshape(2,3)
  }

  "An FCL forward-backward pass" should "return the gradients and the current cost of the batch" in {

    propd._1 shouldEqual vec(-2, 18, -6, 62).reshape(2,2)
    propd._2 shouldEqual List(vec(1, 6, 15, 1, 18, 37, 2, 24, 52).reshape(3,3))
    propd._3 shouldEqual 87
  }

  "An FCL" should "calculate correct gradients with regards to x" in {
    val epsilon = 1e-4
    val x = vec(1, 2, 3, 4).reshape(2, 2)
    val dx1 = vec(0, epsilon, 0, 0).reshape(2, 2)
    val y_bar = vec(2, 6, 5, 3, 4, 6).reshape(2, 3)

    val output = Euclidean()

    val f = Dense(theta) !! output

    val grad = (f.fwbw(x + dx1, y_bar)._3 - f.fwbw(x - dx1, y_bar)._3) / 2 / epsilon
    val grad1 = f.fwbw(x, y_bar)._1
    math.abs(grad - grad1(0,1))/grad should be < math.sqrt(epsilon)
  }

  "An FCL" should "calculate correct gradients with regards to theta" in {
    val epsilon = 1e-4
    val x = vec(1, 2, 3, 4).reshape(2, 2)
    val output = Euclidean()

    val y_bar = vec(2, 6, 5, 3, 4, 6).reshape(2, 3)

    val fcl = Dense(theta)
    val f = fcl !! output
    fcl.update(Seq(vec(0,0,0, 0, epsilon, 0, 0, 0, 0).reshape(3,3)))
    val c_right = f.fwbw(x, y_bar)._3
    fcl.update(Seq(vec(0,0,0, 0, -2 * epsilon, 0, 0, 0, 0).reshape(3,3)))
    val c_left = f.fwbw(x, y_bar)._3

    val grad = (c_right - c_left) / 2 / epsilon
    val grad1 = f.fwbw(x, y_bar)._2.head
    math.abs(grad - grad1(1,1))/grad should be < math.sqrt(epsilon)
  }

  "appf" should "apply arbitrary functions element-wise" in {
    appf(vec(1,2,3,4), _+1) shouldEqual vec(2, 3, 4, 5)
  }

  "A RELU" should "work as specified" in {
    val net = ReLU() !! Euclidean()
    val x = vec(1,-2, 2, 0)
    val y_bar = appf(x, _+2)
    val y = net.ffwd(x)
    y shouldEqual vec(1f, 0, 2, 0)
    val (delta, _, c) = net.fwbw(x, y_bar)
    delta shouldEqual vec(-2, 0, -2, 0)
    c shouldEqual 6

  }
}
