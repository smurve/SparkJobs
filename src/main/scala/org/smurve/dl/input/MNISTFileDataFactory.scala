package org.smurve.dl.input

import java.io.FileInputStream
import java.nio.file.FileSystemException
import java.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.mnist.MNISTImage
import org.smurve.nd4s.toArray


class MNISTFileDataFactory(basePath: String) extends DataFactory {

  private val NUM_TRAINING_PER_FILE = 60000
  private val NUM_TEST_PER_FILE = 10000
  private val IMG_HEIGHT = 28
  private val IMG_WIDTH = 28
  private val IMG_SIZE = IMG_HEIGHT * IMG_WIDTH
  private val IMG_HEADER_SIZE = 16
  private val LBL_HEADER_SIZE = 8

  private val trImgs = readImages("train", NUM_TRAINING_PER_FILE)
  private val teImgs = readImages("test", NUM_TEST_PER_FILE)

  private val trLbls = readLabels("train-labels", NUM_TRAINING_PER_FILE)
  private val teLbls = readLabels("test-labels", NUM_TEST_PER_FILE)


  private def readImages(fileName: String, nRecords: Int ): INDArray = {

    val headerBuffer = new Array[Byte](IMG_HEADER_SIZE)
    val buffer = new Array[Byte](nRecords * IMG_SIZE )

    val stream = new FileInputStream(s"$basePath/$fileName")
    val nh = stream.read(headerBuffer)
    if ( nh != IMG_HEADER_SIZE )
      throw new FileSystemException("Failed to read image header")

    val nb = stream.read(buffer)
    if ( nb != buffer.length )
      throw new FileSystemException("Failed to read images")
    Nd4j.create(buffer.map(p => (p & 0XFF) / 256f )).reshape(nRecords, 1, IMG_WIDTH, IMG_HEIGHT)
  }


  private def readLabels(fileName: String, nRecords: Int ): INDArray = {

    val headerBuffer = new Array[Byte](LBL_HEADER_SIZE)
    val buffer = new Array[Byte](nRecords )
    val lblBuffer10 = new Array[Float](nRecords * 10 )

    val stream = new FileInputStream(s"$basePath/$fileName")
    val nh = stream.read(headerBuffer)
    if ( nh != LBL_HEADER_SIZE )
      throw new FileSystemException("Failed to read label header")

    val nb = stream.read(buffer)
    if ( nb != buffer.length )
      throw new FileSystemException("Failed to read labels")

    for ( i <- 0 until nRecords ) {
      lblBuffer10(10 * i + buffer(i) % 0xFF) = 1f
    }

    Nd4j.create(lblBuffer10).reshape(nRecords, 10)
  }

  override def labelNames: util.List[String] = util.Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

  /**
    * @return the chunk extracted from the next file if there still is one, begin again with the first file, if not.
    */
  override def nextTrainingChunk(): (INDArray, INDArray) =
    (trImgs, trLbls)



  override def hasMoreTraining: Boolean = true

  override def readTestData(): (INDArray, INDArray) =
    (teImgs, teLbls)

  override def startOver(): Unit = ()

  def asImageString(iNDArray: INDArray): String = {
    val arr = toArray(iNDArray)
    MNISTImage(arr.map(d => (d * 256) .toByte), 28, 28).toString
  }


}
