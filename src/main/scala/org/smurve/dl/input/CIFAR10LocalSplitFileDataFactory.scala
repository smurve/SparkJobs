package org.smurve.dl.input

import java.io.{File, FileInputStream, FileNotFoundException}
import java.util

import grizzled.slf4j.Logging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * This data factory assumes that files with the naming pattern
  * lbl_ + FILENAME, img_ + FILENAME can be found in basePath. Obviously these files are assumed
  * to contain byte-encoded images and single-byte encoded labels
  *
  * @param basePath      the directory to find the data files
  * @param trainingFiles An array of the file base names
  * @param testFile      The base name of the test images and labels
  */
class CIFAR10LocalSplitFileDataFactory(basePath: String,
                                       trainingFiles: Array[String],
                                       testFile: String) extends DataFactory with Logging {

  import CIFAR10LocalSplitFileDataFactory._

  override def labelNames: util.List[String] = util.Arrays.asList(categories: _*)

  var cursor: Int = -1

  /**
    * @return the chunk extracted from the next file if there still is one, begin again with the first file, if not.
    */
  override def nextTrainingChunk(): (INDArray, INDArray) = {
    cursor += 1
    val currentTrainingFile = trainingFiles(cursor)
    readSplit(currentTrainingFile)
  }

  override def startOver(): Unit = cursor = -1

  /**
    * read images and labels from their respective (split from original) file
    *
    * @param fileName the base name of the file to read from
    * @return
    */
  private def readSplit(fileName: String, num_records: Int = NUM_RECORDS_PER_FILE): (INDArray, INDArray) = {

    val img_bytes = new Array[Byte](num_records * IMG_SIZE)

    val img_fis = try {
      new FileInputStream(new File(basePath + "/img_" + fileName))
    } catch {
      case _: FileNotFoundException =>
        val missing = basePath + "/img_" + fileName
        error("Required files not found. Download https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
        error("Unzip it into input/cifar10, then run CIFAR10DataSplitter to split them into images and labels.")
        throw new RuntimeException(s"can't continue. File $missing not found.")
    }
    img_fis.read(img_bytes)

    val arr = img_bytes.map(b => (b & 0xFF).toFloat) //  / 256f - 0.5
    val images = Nd4j.create(arr).reshape(num_records, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    images.divi(256f).subi(.5f)

    val lbl_bytes = new Array[Byte](num_records)
    val lbl_fis = new FileInputStream(new File(basePath + "/lbl_" + fileName))
    lbl_fis.read(lbl_bytes)
    val labels = Nd4j.zeros(num_records * 10).reshape(num_records, 10)
    for (i <- 0 until num_records) {
      val value = lbl_bytes(i).toInt
      labels(i, value) = 1.0
    }
    (images, labels)
  }

  /**
    * @return false, if the last file has just been read. Next request will succeed, however, and return the first file
    */
  override def hasMoreTraining: Boolean = cursor + 1 < trainingFiles.length


  override def readTestData(): (INDArray, INDArray) = readSplit(testFile)
}


object CIFAR10LocalSplitFileDataFactory {
  val IMG_WIDTH = 32
  val IMG_HEIGHT = 32
  val NUM_CHANNELS = 3
  val IMG_SIZE: Int = NUM_CHANNELS * IMG_WIDTH * IMG_HEIGHT
  val CHANNEL_SIZE: Int = IMG_WIDTH * IMG_HEIGHT
  val BUFFER_SIZE_PER_ENTRY: Int = 1 + NUM_CHANNELS * CHANNEL_SIZE
  val NUM_RECORDS_PER_FILE = 10000
  val NUM_FILES = 5
  val categories = Array(
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck")

  def label(output: INDArray): String = {
    var index_with_max = 0
    for (index <- 0 until output.length()) {
      if (output.getDouble(index) > output.getDouble(index_with_max))
        index_with_max = index
    }
    categories(index_with_max)
  }

}
