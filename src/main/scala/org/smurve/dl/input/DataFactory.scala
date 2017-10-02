package org.smurve.dl.input

import java.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, ViewIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

trait DataFactory {

  def labelNames: util.List[String]

  def nextTrainingIterator(batchSize: Int, maxRecords: Int = Integer.MAX_VALUE): DataSetIterator = {
    val (images, labels) = nextTrainingChunk()
    new ViewIterator(new DataSet(images, labels), batchSize){
      override def getLabels: util.List[String] = labelNames

      override def hasNext: Boolean = super.hasNext && cursor() < maxRecords
    }
  }

  def testIterator(batchSize: Int): DataSetIterator = {
    val (images, labels) = readTestData()
    new ViewIterator(new DataSet(images, labels), batchSize){
      override def getLabels: util.List[String] = labelNames
    }
  }

  def nextTrainingChunk(): (INDArray, INDArray)
  def hasMoreTraining: Boolean

  def readTestData(): (INDArray, INDArray)

  def startOver(): Unit
}
