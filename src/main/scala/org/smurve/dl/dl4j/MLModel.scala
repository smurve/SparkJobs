package org.smurve.dl.dl4j

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

/**
  * common interface for ParallelWrapper and MultiLayerNetwork, respective adapters for both will implement this
  */
trait MLModel  {

  def underlying() : MultiLayerNetwork

  def fit(dataSetIterator: DataSetIterator)

  def evaluate(dataSetIterator: DataSetIterator): Evaluation
  
}
