package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.language.postfixOps

/**
  * Convolutional Layer
  * Note that we don't consider padding here
  *
  * @param theta        the weight matrix of size D x H x W, the first row must contain the bias in column 0
  * @param depth_input  the number of channels expected in the input vector
  * @param height_input the number of rows of the input vector
  * @param width_input  the number of columns of the input vector
  */
case class Conv(theta: INDArray, depth_input: Int, height_input: Int, width_input: Int,
                height_theta: Int) extends Layer with ParameterSupport with StatsSupport {

  require(theta.rank == 3, "Expecting weight tensor of rank 3")
  require(theta.size(1) - 1 == height_theta, "Theta LRFs need an additional row for the bias. Did you supply?")

  val depth_theta: Int = theta.size(0)
  val width_theta: Int = theta.size(2)

  val depth_output: Int = depth_input * depth_theta
  val height_output: Int = height_input - height_theta + 1 // theta has an additional row for the bias
  val width_output: Int = width_input - width_theta + 1

  /**
    * Multi-Indices. Keeps more-dimensional iterations concise, still readable
    */
  private val indices_output = multiIndex(0 until depth_output, 0 until height_output, 0 until width_output)
  private val indices_theta = multiIndex(0 until depth_theta, 0 until height_theta + 1, 0 until width_theta)
  private val indices_input = multiIndex(0 until depth_input, 0 until height_input, 0 until width_input)

  /**
    * forward pass and back prop in one go.
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    * @return dC/dx for the backprop, the list of all gradients, and the cost
    */
  override def fwbw(x: INDArray, y_bar: INDArray): BackPack = {
    val (dC_dy, grads, cost) = nextLayer.fwbw(fun(x), y_bar)
    (dC_dx(x, dC_dy), dC_dTheta(x, dC_dy) :: grads, cost)
  }

  /**
    * update from head and pass the tail on to subsequent layers
    *
    * @param steps : The list of gradients accumulated during training
    */
  override def update(steps: Seq[INDArray]): Unit = {
    if (booleanParam("print.stats").getOrElse(false)) {
      printStats(name= s"Layer $seqno: ${getClass.getSimpleName}", theta = theta, steps = steps.head)
    }
    theta += steps.head
    nextLayer.update(steps.tail)
  }



  /**
    * The convolution function. If you consider the given naming convention, the logic should be fairly obvious
    * o = ouput, i = input, t = theta
    * d = depth, c = column, r = row
    *
    * @param inp the input vector of rank 3: D_input x H x W
    * @return the output of the convolution: rank 4! D_theta x D_input x H x W
    */
  def fun(inp: INDArray): INDArray = {

    require(inp.rank == 4, "Expecting rank 4: N x D x H x W")

    val N_inp = inp.size(0)
    val output: INDArray = Nd4j.zeros(N_inp, depth_output, height_output, width_output)

    /* this iterates over the entire output tensor using multi-index*/
    for {
      (od, or, oc) <- indices_output
      n <- 0 until N_inp
    } {
      output(n, od, or, oc) = {
        /* this iterates over the weights (local receptive field) */
        val elems =
          for {tr <- 1 until height_theta + 1
               tc <- 0 until width_theta
          } yield {
            val (id, ir, ic) = idrc(or, od, oc, tr, tc)
            val xdrc = inp(n, id, ir, ic)
            val tdrc = theta(td_od(od), tr, tc)
            xdrc * tdrc
          }
        elems.sum + theta(td_od(od), 0, 0)
      }
    }
    val result = output.reshape(N_inp, depth_theta, depth_input, height_output, width_output)

    printOutput(result)

    result
  }

  def numOutputVectors: Int = integerParam("print.output").getOrElse(0)


  def printOutput(array: INDArray): Unit = {
    val n = numOutputVectors
    if ( n > 0 ) {
      for (i <- 0 until n ) {
        val s = for {
          td <- 0 until array.size(1)
          id <- 0 until array.size(2)
        } yield {
          visualize(array(i, td, id, ->, ->).reshape(array.size(3), array.size(4)))
        }
        println(in_a_row(" | ")(s: _*))
      }
    }
  }

  /**
    * The gradient with respect to theta
    * @param x the input vector
    * @param dC_dy the 'delta' from the subsequent layer
    * @return the current gradient
    */
  def dC_dTheta(x: INDArray, dC_dy: INDArray): INDArray = {
    val N_inp = x.size(0)
    val grad = Nd4j.zeros(theta.shape(): _*)

    for {
      (td, tr, tc) <- indices_theta
    } {
      val contrib: Double = (for {
        n <- 0 until N_inp
        or <- 0 until height_output
        oc <- 0 until width_output
        id <- 0 until depth_input
      } yield {
        val od = td * depth_input + id
        val dcdy = dC_dy(n, od / depth_input, od % depth_input, or, oc)

        if ( tr == 0 ) {
          // theta has the bias coefficient in its first row/first column
          if ( tc == 0 )
            dcdy
          else
            0 // other columns in first row are 0
        } else {
          val ir = or + tr - 1
          val ic = oc + tc
          val dcdy = dC_dy(n, od / depth_input, od % depth_input, or, oc)
          val xndrc = x(n, id, ir, ic)
          dcdy * xndrc
        }
      }).sum
      grad (td, tr, tc) = contrib / N_inp
    }
    grad
  }

  /**
    * the gradient with respect to the input vector
    * @param inp the input vector
    * @param dC_dy the 'delta' from the subsequent layer
    * @return the gradient dC/dx
    */
  def dC_dx(inp: INDArray, dC_dy: INDArray): INDArray = {

    val res = Nd4j.zeros(inp.shape(): _*)

    for {
      n <- 0 until inp.size(0)
      (id, ir, ic) <- indices_input
    } {
      val contrib: Double = (for {
        td <- 0 until depth_theta
        or <- math.max(0, ir - height_theta + 1) to math.min(height_output - 1, ir)
        oc <- math.max(0, ic - width_theta + 1) to math.min(width_output - 1, ic)
      } yield {
        val od = depth_input * td + id
        val tr = ir - or
        val tc = ic - oc
        val t_drc = theta(td, tr + 1, tc)
        val dC_dy_drc = dC_dy(n, od / depth_input, od % depth_input, or, oc)
        t_drc * dC_dy_drc
      }).sum

      res(n, id, ir, ic) = contrib
    }
    res
  }


  /** current depth index of the input vector */
  def id_od(od: Int): Int = od % depth_input

  /** current depth index of the weight theta */
  def td_od(od: Int): Int = od / depth_input

  /** current row of input vector as a function of the rows of theta and output */
  def ir_or_tr(or: Int, tr: Int): Int = or + tr - 1

  /** current column of input vector as a function of the columns of theta and output */
  def ic_oc_tc(oc: Int, tc: Int): Int = oc + tc

  /** all input indices depending on the other indices */
  def idrc(or: Int, od: Int, oc: Int, tr: Int, tc: Int): (Int, Int, Int) = (id_od(od), ir_or_tr(or, tr), ic_oc_tc(oc, tc))


}

