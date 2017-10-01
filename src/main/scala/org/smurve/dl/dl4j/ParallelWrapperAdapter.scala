package org.smurve.dl.dl4j

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.parallelism.ParallelWrapper
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator


class ParallelWrapperAdapter(wrapper: ParallelWrapper, val underlying: MultiLayerNetwork) extends MLModel {

  override def fit(iterator: DataSetIterator): Unit = wrapper.fit(iterator)

  override def evaluate(iterator: DataSetIterator): Evaluation = underlying.evaluate(iterator)
}
