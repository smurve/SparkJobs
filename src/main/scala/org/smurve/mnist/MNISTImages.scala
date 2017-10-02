package org.smurve.mnist

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.immutable.IndexedSeq


/**
  * A convenience wrapper around the content of yann's file containing MNIST Images
  * see: http://yann.lecun.com/exdb/mnist/
  */
class MNISTImages( val bytes: Array[Byte] ) extends MNISTData with Serializable {

  val headerSize = 16

  assert(magicNumber == IMAGES, "Magic number does not indicate an image file.")

  val numImgs: Int = asInt(bytes.slice(4,8))
  val numRows: Int = asInt(bytes.slice(8,12))
  val numCols: Int = asInt(bytes.slice(12,16))

  val imgSize: Int = numRows * numCols

  lazy val asByteArray: Array[Byte] = bytes.slice(headerSize, bytes.length)
  lazy val asSequence: IndexedSeq[MNISTImage] = ( 0 until numImgs).map(imgAtIndex)
  lazy val asFloatArray: Array[Float] = asByteArray.map(b=>(b & 0xFF) / 256f)
  lazy val asINDarray: INDArray = Nd4j.create(asFloatArray).reshape(numImgs, imgSize)

  def bytesAtIndex(index: Int): Array[Byte] = bytes.slice( headerSize + imgSize * index, headerSize + imgSize * ( index + 1 ))

  def iNDarrayAtIndex (index: Int): INDArray = Nd4j.create(bytesAtIndex(index).map(b=>(b & 0xFF) / 256f))

  def imgAtIndex (index: Int ) : MNISTImage =
    MNISTImage(bytesAtIndex ( index ), numCols, numRows)

}

