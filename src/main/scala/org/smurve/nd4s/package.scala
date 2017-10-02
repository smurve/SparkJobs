package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.util.Random

/**
  * Created by wgiersche on 25/07/17.
  */
package object nd4s {

  /**
    * the return value of fwbw as a tuple consisting of
    * _1: dC/dX of the layer returning this = dC/dy of the receiving layer. This is the actual back prop term
    * _2: List of all dC/dTheta gradients of the subsequent layers. Prepend your grad to bevor returning from fwbw
    * _3: the current cost
    */
  type BackPack = (INDArray, List[INDArray], Double)

  /**
    * vertically "pad" with ones.
    *
    * 1  1
    *
    * a  b         a  b
    * =>
    * c  d         c  d
    *
    *
    * @param x the input
    * @return the input matrix, padded with ones
    */
  def v1(x: INDArray): INDArray = Nd4j.vstack(Nd4j.ones(x.size(1)), x)

  /**
    * horizontically "pad" with ones.
    *
    * a  b         1  a  b
    * =>
    * c  d         1  c  d
    *
    *
    * @param x the input
    * @return the input matrix, padded with ones
    */
  def h1(x: INDArray): INDArray = Nd4j.hstack(Nd4j.ones(x.size(0)).T, x)

  /**
    * convenience vector literals
    *
    * @param arr the numbers to make up the INDArray
    * @return the INDArray containing those numbers
    */
  def vec(arr: Double*): INDArray = Nd4j.create(Array(arr: _*))


  /**
    * work-around for broken map() on INDArray
    * Only supporting tensors of rank 1 to 5
    */
  def appf(x: INDArray, f: Double => Double): INDArray = {
    val res: INDArray = Nd4j.zeros(x.shape: _*)
    iArr(x).par.foreach { i =>
      i.length match {
        case 1 => res(i(0)) = f(x(i: _*))
        case 2 => res(i(0), i(1)) = f(x(i: _*))
        case 3 => res(i(0), i(1), i(2)) = f(x(i: _*))
        case 4 => res(i(0), i(1), i(2), i(3)) = f(x(i: _*))
        case 5 => res(i(0), i(1), i(2), i(3), i(4)) = f(x(i: _*))
      }
    }
    res
  }


  /**
    *
    * @param labeledData the images/labels to be shuffled
    * @return
    */
  def shuffle(labeledData: (INDArray, INDArray), random: Random = new Random()): (INDArray, INDArray) = {

    require(labeledData._1.size(0) == labeledData._2.size(0), "Arrays to shuffle should have identical length")

    val (samples, labels) = labeledData
    val combined = Nd4j.hstack(samples, labels)

    Nd4j.shuffle(combined, random.self, 1)

    val lc = combined(->, 0 -> samples.size(1))
    val rc = combined(->, samples.size(1) -> combined.size(1))
    (lc, rc)
  }

  def toArray(iNDArray: INDArray): Array[Double] = {
    val length = iNDArray.length
    val array = Array.fill(length) {
      0.0
    }

    array.indices.foreach { index =>
      array(index) = iNDArray.getDouble(index)
    }
    array
  }

  def equiv10(classification: INDArray, label: INDArray): Boolean = {
    val max = classification.max(1).getDouble(0)
    val lbl_icx = (label ** vec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9).T).getDouble(0).toInt
    val nPreds = (classification === max).sum(1).getFloat(0)
    val res = nPreds == 1 && max == classification.getDouble(lbl_icx)
    res
  }


  /**
    * Index Arrays for convenient access to INDArray tensors
    */
  type IArr = Array[Array[Int]]

  /**
    * @return the "outer product": the Array containing all combinations
    */
  def outer(l: IArr, r: IArr): IArr = l.flatMap(i => r.map(i ++ _))

  /**
    * @return an index array from ranges
    */
  def iArr(ranges: Range*): IArr = {
    ranges.map(r => r.toArray.map(Array(_))).toArray.reduce(outer)
  }

  /**
    * @return all indices of a given INDArray
    */
  def iArr(v: INDArray): IArr = iArr(v.shape().map(0 until _): _*)

  /**
    * convenience method for displaying index arrays
    */
  def asString(arr: IArr): String = arr.toList.map(_.toList).toString
    .replace("List", "").replace("(", "[").replace(")", "]")


  def reduceByElimination(source: INDArray, mi: IArr, discriminator: (Double, Double) => Boolean): (Double, Array[Int]) = {
    mi.map(i => (source(i: _*), i))
      .reduce((a, e) => if (discriminator(a._1, e._1)) a else e)
  }

  def minWithIndex(source: INDArray, mi: IArr): (Double, Array[Int]) = reduceByElimination(source, mi, _ < _)

  def maxWithIndex(source: INDArray, mi: IArr): (Double, Array[Int]) = reduceByElimination(source, mi, _ > _)


  /**
    * Utilty function to create multi-index Seq e.g. for iterating over entire tensors
    * create a cross-product-like Seq of triples from three Seqs of Ints
    *
    * @param s1 the outermost sequence
    * @param s2 the middle sequence
    * @param s3 the innermost sequence
    * @return a Sequence of triples
    */
  def multiIndex(s1: Seq[Int], s2: Seq[Int], s3: Seq[Int]): Seq[(Int, Int, Int)] =
    s1.flatMap(i => s2.map(j => (i, j)))
      .flatMap(p => s3.map(k => (p._1, p._2, k)))


  private def scaleToByte(min: Double, max: Double)(x: Double): Byte = {
    if (x < 0) 0 else {
      val min0 = math.max(0, min)
      (255 * (x - min0) / (max - min0)).toByte
    }
  }


  def visualize(x: INDArray): String = {
    val hborder = " " + ("-" * 2 * x.size(0))
    require(x.rank == 2, "Can only visualize 2-dim arrays")
    val min: Double = x.minT[Double]
    val max: Double = x.maxT[Double]
    val img = (0 until x.size(0)).map(i => {
      val arr = toArray(x(i, ->))
      val row = arr.map(scaleToByte(min, max))
      rowAsString(row)
    }).mkString("\n")
    hborder + "\n" + img + "\n" + hborder
  }


  private def rowAsString(bytes: Array[Byte]): String = {
    val res = bytes.map(b => {
      val n = b & 0xFF
      val c = if (n == 0) 0 else n / 32 + 1
      c match {
        case 0 => "  "
        case 1 => "' "
        case 2 => "''"
        case 3 => "::"
        case 4 => ";;"
        case 5 => "cc"
        case 6 => "OO"
        case 7 => "00"
        case 8 => "@@"
      }
    }).mkString("")
    "|" + res + "|"
  }

  /**
    * @param sep  the separator to display between each two images
    * @param imgs any number of equally-sized multi-line strings (using '\n')
    * @return a string showing all multi-line strings as images in a row
    */
  def in_a_row(sep: String = " ")(imgs: String*): String = {
    imgs.map(_.split("\n").toList).reduce((l, r) => {
      l.zip(r).map(p => p._1 + sep + p._2)
    }).reduce((l, r) => l + "\n" + r) + "\n"
  }


}
