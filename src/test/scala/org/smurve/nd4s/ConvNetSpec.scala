package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, ShouldMatchers}

class ConvNetSpec extends FlatSpec with ShouldMatchers with TestTools {

  trait TestData {
    val x: INDArray = vec(
      0, 1, 2, 3, 4,
      1, 2, 3, 4, 5,
      2, 3, 4, 5, 6,
      3, 4, 5, 6, 7,
      4, 5, 6, 7, 8,

      4, 5, 6, 7, 8,
      3, 4, 5, 6, 7,
      2, 3, 4, 5, 6,
      1, 2, 3, 4, 5,
      0, 1, 2, 3, 4
    ).reshape(1, 2, 5, 5)

    val theta1: INDArray = vec(
      0, 0,
      1, 1,
      -1, -1,

      0, 0,
      -2, 2,
      -2, 2,

      0, 0,
      1, -.5,
      -.5, 1
    ).reshape(3, 3, 2)

    val expected_conv_result: INDArray = vec(
      -2, -2, -2, -2,
      -2, -2, -2, -2,
      -2, -2, -2, -2,
      -2, -2, -2, -2,

      2, 2, 2, 2,
      2, 2, 2, 2,
      2, 2, 2, 2,
      2, 2, 2, 2,

      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,

      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,

      1, 2, 3, 4,
      2, 3, 4, 5,
      3, 4, 5, 6,
      4, 5, 6, 7,

      4, 5, 6, 7,
      3, 4, 5, 6,
      2, 3, 4, 5,
      1, 2, 3, 4
    ).reshape(1, 3, 2, 4, 4)

    // happens to have the same shape, of course
    val fake_dC_dy: INDArray = expected_conv_result

    val expected_pool_result: INDArray = vec(
      0, 0,
      0, 0,

      4, 4,
      4, 4,

      3, 5,
      3, 5
    ).reshape(1, 3, 2, 2)

    val theta2: INDArray = vec(
      0, 0,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1
    ).reshape(13, 2)

    val conv = Conv(theta1, depth_input = 2, height_input = 5, width_input = 5, height_theta = 2)
    val pool: Layer = (
      AvgPool(depth_stride = 2, height_stride = 2, width_stride = 2)
        !! Flatten(3, 2, 2)).asInstanceOf[AvgPool]
    val dense = Dense(theta2)
    val output = Euclidean()
  }


  "A euclidean output layer" should "compute the L2 norm of the difference" in {
    new TestData {
      val propd: BackPack = output.fwbw(vec(1, 2), vec(0, 0))
      propd._3 shouldEqual 2.5
      val dC_dx: INDArray = propd._1
      dC_dx shouldEqual vec(1, 2)
    }
  }


  "A conv layer" should "produce 6 feature matrices for 2 channels and 3 LRFs" in {
    new TestData {
      val y: INDArray = conv.fun(x)
      y shouldEqual expected_conv_result

    }
  }

  "A conv net" should "compute dC/dx" in {
    new TestData {
      conv !! output
      conv.dC_dx(x, fake_dC_dy)
    }
  }

  "A conv layer" should "compute dC/dx correctly" in {

    new TestData {
      val dC_dx: INDArray = conv.dC_dx(x, fake_dC_dy)
    }
  }

  "A conv net" should "build from basic layers and compute output vectors correctly" in {

    new TestData {
      val convnet: Layer = conv !! pool !! dense !! output

      val y: INDArray = convnet.ffwd(x)

      y shouldEqual vec(32, -32)
    }
  }


  "A conv net" should "compute backprop dC/dx and gradient correctly" in {
    new TestData {
      val convnet: Layer = conv !! pool !! dense !! output
      val y_bar: INDArray = vec(30, -32).reshape(1, 2)

      validateBackProp(convnet, x, y_bar)
      validateGradTheta(convnet, theta2, 1, x, y_bar)
      validateGradTheta(convnet, theta1, 0, x, y_bar)
    }
  }

  it should "exhibit certain symmetries." in {
    new TestData {
      val convnet: Layer = conv !! pool !! dense !! output
      val x2: INDArray = Nd4j.vstack(x,x)

      val y_bar2: INDArray = vec(30, -32, 31, -31).reshape(2, 2)
      checkSymmetries(conv, x2, y_bar2)
    }
  }
}
