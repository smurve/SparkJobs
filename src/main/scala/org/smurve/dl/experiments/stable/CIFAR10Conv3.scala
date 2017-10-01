package org.smurve.dl.experiments.stable

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4s.Implicits._
import org.smurve.dl.dl4j.{MLModel, WebUI}
import org.smurve.dl.input.CIFAR10LocalSplitFileDataFactory._
import org.smurve.dl.input.{CIFAR10LocalSplitFileDataFactory, DataFactory}
import org.smurve.dl.models.Conv3ModelBuilder
import org.smurve.nd4s.visualize
import scopt.OptionParser

/**
  * Experiment:
  * Learn to classify CIFAR-10 images with a 3 Layer Conv Net
  */
object CIFAR10Conv3 {

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF)
  case class DefaultParams(
                            seed: Int = 12345,
                            eta: Double = 2e-1,
                            n_files: Int = 5,
                            n_epochs: Int = 1,
                            size_batch: Int = 100,
                            n_tests: Int = 1000,
                            nc1: Int = 16,
                            nc2: Int = 32,
                            nc3: Int = 64,
                            n_dense: Int = 512,
                            updater: Updater = Updater.ADAM,
                            parallel: Int = 6
                          )



  def main(args: Array[String]): Unit = {

    val params = determineParams(args, DefaultParams())
    val model = modelFromParams(params)

    val webserver = new WebUI(model.underlying()).uiServer

    val trainingFiles = (1 to NUM_FILES).map(n => s"data_batch_$n.bin").toArray
    val chunkSize = NUM_RECORDS_PER_FILE
    val dataFactory = new CIFAR10LocalSplitFileDataFactory("./input/cifar10", trainingFiles, "test_batch.bin")
    val testData = dataFactory.testIterator(batchSize = params.n_tests)

    for (epoch <- 1 to params.n_epochs) {

      println(s"\nStarting epoch Nr. $epoch")
      var fileNr = 1

      // This data factory loads chunks from file, as they are needed
      while ( dataFactory.hasMoreTraining ) {

        println(s"Reading from file $fileNr")
        val trainingData = dataFactory.nextTrainingIterator(params.size_batch)
        println(s"Starting training with file $fileNr"); fileNr += 1

        val startAt = System.currentTimeMillis()

        model.fit(trainingData)
        val finishAt = System.currentTimeMillis()

        val eval = model.evaluate(testData)

        println(eval.stats)
        println(s"$chunkSize samples learned after ${((finishAt - startAt) / 100) / 10.0} seconds.")

      }
      dataFactory.startOver()
    }

    //inferenceDemo(model, dataFactory, params, 10)

    //printParams(model, "0_W", "0_b")

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
    * @param testDataFactory the data factory that produces the test set
    * @param num_records the number of records to classify
    */
  def inferenceDemo(model: MultiLayerNetwork, testDataFactory: DataFactory, params: DefaultParams, num_records: Int): Unit = {
    val testSet = testDataFactory.nextTrainingChunk()._1

    (0 until 10).foreach { i =>

      val image = (0 until NUM_CHANNELS).map(c => testSet(i, c, ->)).reduce(_ + _)

      println(visualize(image))

      val input = testSet(i, ->).reshape(1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
      val prediction = model.output(input)
      println(prediction.toString + ": " + CIFAR10LocalSplitFileDataFactory.label(prediction))
    }

  }

  /**
    * determine parameters from defaults and command line params
    */
  def determineParams(args: Array[String], defaults: DefaultParams): DefaultParams = {

    val parser: OptionParser[DefaultParams] = new OptionParser[DefaultParams]("CIFAR10Runner") {
      head("CIFAR10Runner", "1.0")

      opt[Int]('E', "n-epochs").valueName("Number of Epochs")
        .action((x, args) => args.copy(n_epochs = x))

      opt[Int]('b', "size-batch").valueName("Mini Batch size")
        .action((x, args) => args.copy(size_batch = x))

      opt[Int]('b', "n-files").valueName("Number of files to use [1-5]")
        .action((x, args) => args.copy(n_files = x))

      opt[Int]('t', "n-tests").valueName("Mini Batch size")
        .action((x, args) => args.copy(n_tests = x))

      opt[Double]('e', "eta").valueName("Learning rate")
        .action((x, args) => args.copy(eta = x))

    }

    parser.parse(args, defaults)
      .getOrElse({
        System.exit(-1) // error message has already been created by the parser
        // Actually, there should be a method in scala that returns Nothing, but nobody ever cared, as it appears
        throw new RuntimeException("Just satisfying the compiler. This won't ever happen.")
      })
  }

  /**
    * create the model from the relevant parameters
    * @param params the given command line parameters
    * @return a model build with those given parameters
    */
  private def modelFromParams(params: DefaultParams): MLModel =
    Conv3ModelBuilder.build(
      seed = params.seed,
      eta = params.eta,
      nc1 = params.nc1,
      nc2 = params.nc2,
      nc3 = params.nc3,
      n_dense = params.n_dense,
      parallel = params.parallel)
}