package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}

import scala.util.Random

class PackageTests extends FlatSpec with ShouldMatchers with TestTools{

  val seed = 123

  "shuffle" should "shuffle - obviously." in {

    val orig1 = (1 to 12).asNDArray(4,3)
    val orig2 = (13 to 24).asNDArray(4,3)
    val shuffled = shuffle((orig1, orig2), new Random(seed))
    (shuffled._1 == orig1) should be (false)
    val expected = shuffled._2 - Nd4j.ones(12).reshape(4,3) * 12
    shuffled._1 shouldEqual expected
  }

  val input: INDArray = vec(
    1,2,3,
    4,5,6,
    7,8,9,

    1,2,0,
    4,10,6,
    7,8,9).reshape(2,3,3)

  "extract()" should "extract max/min values correctly " in {

    reduceByElimination(input, iArr(input), _>_)._1 shouldEqual 10.0
    reduceByElimination(input, iArr(input), _>_)._2 shouldEqual Array(1,1,1)

    reduceByElimination(input, iArr(input), _<_)._1 shouldEqual 0.0
    reduceByElimination(input, iArr(input), _<_)._2 shouldEqual Array(1,0,2)

    reduceByElimination(input.ravel, iArr(input.ravel), _>_)._1 shouldEqual 10.0
    reduceByElimination(input.ravel, iArr(input.ravel), _>_)._2 shouldEqual Array(0, 13)

  }

  "visualize()" should "create image strings from arbitrary INDarrays" in {

    val img = visualize(vec(0,1,2,3).reshape(2,2))

    img shouldEqual " ----\n|  ::|\n|OO@@|\n ----"


    //val hor = in_a_row("-")(img, img)
    //hor shouldEqual " ---- - ----\n|  ::|-|  ::|\n|OO@@|-|OO@@|\n ---- - ----"
  }

}
