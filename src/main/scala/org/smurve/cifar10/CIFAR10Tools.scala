package org.smurve.cifar10

import java.io.DataInputStream

import com.sksamuel.scrimage.{Image, RGBColor}
import org.nd4j.linalg.api.ndarray.INDArray

import scala.language.postfixOps

trait CIFAR10Tools {

  val IMG_WIDTH = 32
  val IMG_HEIGHT = 32
  val NUM_CHANNELS = 3
  val CHANNEL_SIZE: Int = IMG_WIDTH * IMG_HEIGHT
  val BUFFER_SIZE_PER_ENTRY: Int = 1 + NUM_CHANNELS * CHANNEL_SIZE
  val NUM_RECORDS_PER_FILE = 10000




  /**
    * read the next image out of an open stream: The structure is assumed to be 1 + 3 x 32 x 32.
    * 1 byte for the label and 3 x 1024 bytes for the three RGB Layers of the image
    *
    * @param stream an open data input stream
    * @return the image and a label
    */
  def nextImage(stream: DataInputStream): (Image, Int) = {
    val buffer = new Array[Byte](BUFFER_SIZE_PER_ENTRY)
    val check = stream.read(buffer)
    assert(check == BUFFER_SIZE_PER_ENTRY, s"Failed to read $BUFFER_SIZE_PER_ENTRY bytes. Got $check instead")
    val pixels = for (pos <- 1 to CHANNEL_SIZE) yield {
      val red = buffer(pos)
      val green = buffer(pos + CHANNEL_SIZE)
      val blue = buffer(pos + 2 * CHANNEL_SIZE)
      RGBColor(red & 0xFF, green & 0xFF, blue & 0xFF).toPixel
    }
    (Image(IMG_WIDTH, IMG_HEIGHT, pixels.toArray), buffer(0).toInt)
  }

  def asImage(inda: INDArray): Image = {
    assert(inda.shape() sameElements Array(1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH))
    val pixels = for (pos <- 0 until CHANNEL_SIZE) yield {
      val red = (inda.getDouble(pos) * 256 ).toInt
      val green = (inda.getDouble(pos + CHANNEL_SIZE) * 256 ).toInt
      val blue = (inda.getDouble(pos + 2 * CHANNEL_SIZE) * 256 ).toInt
      RGBColor(red, green, blue ).toPixel
    }
    Image(IMG_WIDTH, IMG_HEIGHT, pixels.toArray)
  }


}
