package org.smurve.dl.experiments.stable

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4s.Implicits._
import org.smurve.dl.dl4j.{MLModel, WebUI}
import org.smurve.dl.input.ShapeDataFactory
import org.smurve.dl.models.Conv3ModelBuilder
import org.smurve.nd4s.visualize
import scopt.OptionParser

/**
  * Learn to classify 10 different shapes in hand-crafted images with a 3 Layer Conv Net
  */
object ShapesConv3 {

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF)
  case class DefaultParams(
                            seed: Int = 12345,
                            eta: Double = 1e-0,
                            n_channels: Int = 3,
                            imgSize: Int = 20,
                            n_symbols: Int = 10,
                            noise: Double = 0.3,
                            n_epochs: Int = 10,
                            size_batch: Int = 100,
                            n_batches: Int = 100,
                            nc1: Int = 10,
                            nc2: Int = 10,
                            nc3: Int = 10,
                            n_dense: Int = 100,
                            updater: Updater = Updater.ADAM,
                            parallel: Int = 4
                          )


  def main(args: Array[String]): Unit = {

    val params = determineParams(args, DefaultParams())

    /** the model */
    val model = modelFromParams(params)
    val webserver = new WebUI(model.underlying()).uiServer

    /** the data */
    val chunkSize = params.size_batch * params.n_batches
    val dataFactory = new ShapeDataFactory(testSize = 1000,
      chunkSize = chunkSize, imgSize = params.imgSize, depth = params.n_channels,
      n_symbols = 10, noise = 0.3, seed = params.seed)

    /** iterators */
    val trainingData = dataFactory.nextTrainingIterator( batchSize = params.n_batches)
    val testData = dataFactory.testIterator(batchSize = 1000)

    /** start training */
    for (epoch <- 1 to params.n_epochs) {

      println(s"\nStarting epoch Nr. $epoch")
      val startAt = System.currentTimeMillis()

      model.fit(trainingData)
      val finishAt = System.currentTimeMillis()

      val eval = model.evaluate(testData)

      println(eval.stats)
      println(s"$chunkSize samples learned after ${((finishAt - startAt) / 100) / 10.0} seconds.")

    }

    inferenceDemo(model.underlying(), dataFactory, params, 10)

    printParams(model.underlying(), "0_W", "0_b")

    println("Done.")


    webserver.stop()
  }




  /**
    * identify parameters by their key. This is Nd4j-specific:
    * The following key works for conv and dense layers: index _ [W|b],
    * e.g. "0_W" for the weight matrix of the very first layer
    *
    * @param model the model to be dissected
    * @param keys  key identifier for the parameters
    */
  def printParams(model: MultiLayerNetwork, keys: String*): Unit = {
    println("Convolutional Layer (0):")
    keys.foreach { key =>
      val paramVector = model.getParam(key)
      println(key)
      println(paramVector)
    }
  }

  /**
    * Demonstrate inference with a couple of newly-generated records
    *
    * @param model       the model to use
    * @param shapeData   the data generator
    * @param num_records the number of records to classify
    */
  def inferenceDemo(model: MultiLayerNetwork, shapeData: ShapeDataFactory, params: DefaultParams, num_records: Int): Unit = {
    val testSet = shapeData.nextTrainingChunk()._1

    (0 until 10).foreach { i =>

      val image = (0 until params.n_channels).map(c => testSet(i, c, ->)).reduce(_ + _)

      println(visualize(image))

      val input = testSet(i, ->).reshape(1, params.n_channels, params.imgSize, params.imgSize)
      val prediction = model.output(input)
      println ( prediction.toString + " - " + ShapeDataFactory.label(prediction))
    }

  }

  /**
    * passes the relevant parameters to the model builder
    * @param params cmd line params relevant for the model
    * @return a new model based on the given parameters
    */
  private def modelFromParams(params: DefaultParams): MLModel = {
    Conv3ModelBuilder.build(
      seed = params.seed,
      eta = params.eta,
      n_classes = params.n_symbols,
      depth = params.n_channels,
      imgSize = params.imgSize,
      nc1 = params.nc1,
      nc2 = params.nc2,
      nc3 = params.nc3,
      n_dense = params.n_dense,
      updater = params.updater
    )
  }

  /**
    * determine hyperparams from defaults and command line params
    */
  def determineParams(args: Array[String], defaults: DefaultParams): DefaultParams = {

    val parser: OptionParser[DefaultParams] = new OptionParser[DefaultParams]("CIFAR10Runner") {
      head("CIFAR10Runner", "1.0")

      opt[Int]('E', "num-epochs").valueName("Number of Epochs")
        .action((x, args) => args.copy(n_epochs = x))

      opt[Int]('t', "num-batches").valueName("Number of batches")
        .action((x, args) => args.copy(n_batches = x))

      opt[Int]('b', "size-batch").valueName("Mini Batch size")
        .action((x, args) => args.copy(size_batch = x))

      opt[Double]('e', "eta").valueName("Learning rate")
        .action((x, args) => args.copy(eta = x))

      opt[Int]('p', "parallel").valueName("parallel threads")
        .action((x, args) => args.copy(parallel = x))
    }

    parser.parse(args, defaults)
      .getOrElse({
        System.exit(-1) // error message has already been created by the parser
        // Actually, there should be a method in scala that returns Nothing, but nobody ever cared, as it appears
        throw new RuntimeException("Just satisfying the compiler. This won't ever happen.")
      })
  }


}