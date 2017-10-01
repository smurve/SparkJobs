package org.smurve.dl.input

import java.io.{File, FileInputStream, FileOutputStream}
import CIFAR10LocalSplitFileDataFactory._

/**
  * Utility to split data into images and labels
  */
object CIFAR10DataSplitter {

  /**
    * split the files into images and labels
    */
  def main(args: Array[String]): Unit = {

    val fileNames = (1 to 5).map(n => s"data_batch_$n.bin").toArray :+ "test_batch.bin"

    fileNames.foreach(fileName => {
      println(s"reading $fileName")

      val orig = new Array[Byte](NUM_RECORDS_PER_FILE * BUFFER_SIZE_PER_ENTRY)
      val imgs = new Array[Byte](NUM_RECORDS_PER_FILE * IMG_SIZE)
      val lbls = new Array[Byte](NUM_RECORDS_PER_FILE)
      val fis = new FileInputStream(new File("input/cifar10/" + fileName))
      fis.read(orig)
      val fosi = new FileOutputStream(new File("input/cifar10/" + "img_" + fileName))
      val fosl = new FileOutputStream(new File("input/cifar10/" + "lbl_" + fileName))

      for (n <- 0 until NUM_RECORDS_PER_FILE) {
        val offset_orig = n * BUFFER_SIZE_PER_ENTRY
        val offset_imag = n * IMG_SIZE
        lbls(n) = orig(offset_orig)
        for (p <- 0 until IMG_SIZE) {
          imgs(offset_imag + p) = orig(offset_orig + p + 1)
        }
      }
      fosi.write(imgs)
      fosi.close()
      fosl.write(lbls)
      fosl.close()

    })
  }


}
