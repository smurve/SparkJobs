package org.smurve.cifar10

import org.apache.spark.SparkContext
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.cifar10.config.CIFAR10Config
import org.smurve.dl.input.CIFAR10LocalSplitFileDataFactory._
import org.smurve.dl.input.CIFAR10Tools

import scala.language.postfixOps

trait CIFAR10SparkEnv extends CIFAR10Tools {

  protected val config: CIFAR10Config
  protected val session: SparkSession
  protected val sc: SparkContext

  /**
    *
    * @param pattern a file pattern to be appended to the config's base path (prefix)
    * @return an RDD containing the file names and the associated portable data streams
    */
  def readData(pattern: String): RDD[(String, PortableDataStream)] = sc.binaryFiles(config.prefix + s"/$pattern")


  /**
    * read an entire data file into a pair on INDArrays
    * This file is assumed to contain 10000 records of size 3073
    *
    * @param fileName the simple file name of the binary file
    * @return
    */
  def read(fileName: String, num_records: Int = NUM_RECORDS_PER_FILE): (INDArray, INDArray) = {
    val rdd = readData(fileName)
    val pds: PortableDataStream = rdd.collect.head._2
    val bytes = pds.toArray()// new Array[Byte](num_records * BUFFER_SIZE_PER_ENTRY)

    val arr = bytes.map(b => (b & 0xFF).toFloat)

    val ndarr = Nd4j.create(arr)
    val asRecords = ndarr.reshape(NUM_RECORDS_PER_FILE, BUFFER_SIZE_PER_ENTRY)
    val reduced = asRecords(0 -> num_records, ->)

    val samples = reduced(->, 1 ->).reshape(num_records, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH) / 255

    val labels = Nd4j.zeros(10 * num_records).reshape(num_records, 10)
    for (i <- 0 until num_records)  {
      val value = bytes(i * BUFFER_SIZE_PER_ENTRY).toInt
      labels(i, value) = 1.0
    }

    (samples, labels)
  }

}
