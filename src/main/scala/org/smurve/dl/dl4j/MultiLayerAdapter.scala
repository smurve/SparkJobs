package org.smurve.dl.dl4j

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class MultiLayerAdapter ( val underlying: MultiLayerNetwork) extends MLModel {

  override def fit(iterator: DataSetIterator): Unit = underlying.fit(iterator)

  override def evaluate(iterator: DataSetIterator): Evaluation = underlying.evaluate(iterator)
}
