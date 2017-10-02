package org.smurve.dl.experiments.stable

import org.deeplearning4j.nn.conf.Updater
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.smurve.dl.dl4j.{MLModel, WebUI}
import org.smurve.dl.input.CIFAR10LocalSplitFileDataFactory
import org.smurve.dl.input.CIFAR10LocalSplitFileDataFactory._
import org.smurve.dl.models.Conv3ModelBuilder
import org.smurve.util.timeFor
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
                            nc1: Int = 16,
                            nc2: Int = 32,
                            nc3: Int = 64,
                            n_dense: Int = 512,
                            updater: Updater = Updater.ADAM,
                            parallel: Int = 10
                          )

  def main(args: Array[String]): Unit = {

    val params = determineParams(args, DefaultParams())
    val model = modelFromParams(params)

    val webserver = new WebUI(model.underlying()).uiServer

    val trainingFiles = (1 to NUM_FILES).map(n => s"data_batch_$n.bin").toArray
    val chunkSize = NUM_RECORDS_PER_FILE
    val dataFactory = new CIFAR10LocalSplitFileDataFactory("./input/cifar10", trainingFiles, "test_batch.bin")
    val testData = dataFactory.testIterator(batchSize = params.size_batch)

    for (epoch <- 1 to params.n_epochs) {

      println(s"\nStarting epoch Nr. $epoch")
      var fileNr = 1

      // This data factory loads chunks from seperate files
      while ( dataFactory.hasMoreTraining ) {

        println(s"Reading from file $fileNr")
        val trainingData = dataFactory.nextTrainingIterator(params.size_batch)
        println(s"Starting training with file $fileNr"); fileNr += 1

        val (_, t) = timeFor(model.fit(trainingData))
        println(s"$chunkSize samples learned after $t seconds.")
        println(s"Evaluating with ${testData.numExamples()} records.")
        val (eval, te) = timeFor(model.evaluate(testData))
        println(s"Evaluation took $te seconds.")

        println(eval.stats)

      }
      dataFactory.startOver()
    }

    println("Done.")


    webserver.stop()
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

      opt[Int]('f', "n-files").valueName("Number of files to use [1-5]")
        .action((x, args) => args.copy(n_files = x))

      opt[Int]('p', "parallel").valueName("Number of parallel workers")
        .action((x, args) => args.copy(parallel = x))

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