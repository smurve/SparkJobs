package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

import scala.collection.immutable.Seq

/**
  * A simple low-level SGD optimizer
  */
class SimpleSGD() {


  /**
    * Train the model using SGD
    * @param model the model to be trained
    * @param nBatches number of batches
    * @param parallel number of parralel execution threads to use
    * @param equiv a function to compare the results with the given labels. Used for validation
    * @param trainingSet the training set - images and labels
    * @param testSet the test set - images and labels
    * @param n_epochs number of epochs to spend training
    * @param eta the *pesky* learning rate
    * @param reportEveryAfterBatches number of batches to pass before reporting cost
    */
  def train(model: Layer, nBatches: Int, parallel: Int =1, equiv: (INDArray, INDArray) => Boolean,
            trainingSet: (INDArray, INDArray), testSet: (INDArray, INDArray),
            n_epochs: Int, eta: Double, reportEveryAfterBatches: Int): Unit = {

    require(parallel >= 1, "Parallelism can't be less than 1.")

    val t0 = System.currentTimeMillis()

    val batchSize = trainingSet._1.size(0) / nBatches

    require(batchSize >= parallel, "batch size must be larger or equal parallelism.")
    val blockSize = batchSize / parallel
    val nBlocks = batchSize / blockSize

    for (epoch <- 1 to n_epochs) {

      println(s"Starting epoch $epoch")

      //println("  shuffling...")
      val (samples, labels) = trainingSet //shuffle(trainingSet, random = random)
      //println("  Done.")

      for (batchNo <- 0 until nBatches) {

        val offset = batchNo * batchSize

        /**
          * Here, we parallelize by mapping each batch to blocks and reducing (summing the gradients) afterwards
          */
        val blocks = if ( parallel > 1 ) (0 until nBlocks).par else 0 until nBlocks

        val (g_total, c_total): (Seq[INDArray], Double) = blocks.map(block => {

            val (sample_block, label_block) = sliceBlock(blockSize, offset, block, samples, labels)

            val (_, grads, c) = model.fwbw(sample_block, label_block)
            (grads, c)

          }).reduce(sumGradAndCostAtOnce)

        model.update(g_total.map(_ * -eta))

        if (batchNo % reportEveryAfterBatches == 0)
          println(s"Cost: $c_total")
      }

      println("validating...")
      validate(model, testSet, equiv)

    }

    val t1 = System.currentTimeMillis()
    println(s"finished training after ${t1 - t0} ms.")

  }

  /** summing grad and cost at once. Used in reduce phase */
  def sumGradAndCostAtOnce(a: (Seq[INDArray], Double), b: (Seq[INDArray], Double)): (Seq[INDArray], Double) = {
    val sumGrad = (a._1 zip b._1) map { case (gradA, gradB) => gradA + gradB }
    val sumCost = a._2 + b._2
    (sumGrad, sumCost)
  }

  /**
    * slice a block from the samples. Note that no data is copied here.
    * @param blockSize the number of images in the block
    * @param offset the offset to start from
    * @param block the current index
    * @param samples the INDarray of samples
    * @param labels the INDArray of labels
    * @return blocks of images and labels
    */
  def sliceBlock ( blockSize: Int, offset: Int, block: Int, samples: INDArray, labels: INDArray ): (INDArray, INDArray) = {
    val fromIndex = offset + block * blockSize
    val toIndex = fromIndex + blockSize

    ( samples(fromIndex -> toIndex, ->), labels(fromIndex -> toIndex, ->))
  }


  /**
    * Validate the model against a given test set
    * @param model the model to validate
    * @param testSet the pair of images/labels to validate against
    */
  def validate ( model: Layer, testSet: (INDArray, INDArray), equiv: (INDArray, INDArray) => Boolean ): Unit = {
    val N_TEST = testSet._2.size(0)
    val res = model.ffwd(testSet._1)

    val success = ( 0 until N_TEST).map( i=>{
      val pred = res(i,->)
      val label = testSet._2(i, ->)
      if ( equiv( pred, label) ) 1.0 else 0.0
    }).sum / N_TEST

    println( s"Success rate: ${(success*1000).toInt/10.0}")

  }

}
