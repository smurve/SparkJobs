package org.smurve.dl.experiments.stable

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.smurve.dl.dl4j.{MLModel, WebUI}
import org.smurve.dl.experiments.stable.NaiveMNISTDenseNetDemo.IMAGE_SIZE
import org.smurve.dl.input.MNISTFileDataFactory
import org.smurve.dl.models.Conv3ModelBuilder
import org.smurve.nd4s._
import org.smurve.transform.Grid
import org.smurve.util.timeFor
import scopt.OptionParser

import scala.util.Random

/**
  * Experiment:
  * Learn to classify MNIST images with a 3 Layer Conv Net
  */
object MNISTConv3 {

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF)

  case class DefaultParams(
                            seed: Int = 12345,
                            eta: Double = 1e-1,
                            n_epochs: Int = 1,
                            size_batch: Int = 100,
                            nc1: Int = 16,
                            nc2: Int = 32,
                            nc3: Int = 64,
                            n_dense: Int = 200,
                            updater: Updater = Updater.ADAM,
                            n_channels: Int = 1,
                            imgSize: Int = 28,
                            parallel: Int = 1,
                            n_train: Int = 6000,
                            n_test: Int = 10000
                          )


  def main(args: Array[String]): Unit = {

    val params = determineParams(args, DefaultParams())
    val model = modelFromParams(params)

    val webserver = new WebUI(model.underlying()).uiServer

    val dataFactory = new MNISTFileDataFactory("input/mnist")
    val testData = dataFactory.testIterator(batchSize = params.size_batch)
    val trainingData = dataFactory.nextTrainingIterator(params.size_batch, params.n_train)


    for (epoch <- 1 to params.n_epochs) {

      println(s"\nStarting epoch Nr. $epoch")

      val (_, t) = timeFor(model.fit(trainingData))
      println(s"${params.n_train} samples learned after $t seconds.")
      println(s"Evaluating with ${testData.numExamples()} records.")
      val (eval, te) = timeFor(model.evaluate(testData))
      println(s"Evaluation took $te seconds.")

      println(eval.stats)
    }

    inferModel(model.underlying(), dataFactory.rawData(params.n_train, params.n_test )._2)

    println("Done.")

    webserver.stop()
  }

  /**
    * Infer from the given samples, printing the first result
    */
  def inferModel(model: MultiLayerNetwork, test: (INDArray, INDArray)): Unit = {

    val ( imgs, lbls ) = test
    val rnd = new Random()

    for (_ <- 0 to 10) {
      val idx = rnd.nextInt(imgs.size(0))
      val sample = imgs(idx, ->).reshape(1, IMAGE_SIZE)

      val res = model.output(sample)

      println(new Grid(sample.reshape(28, 28)))
      val labeledAs = (lbls(idx, ->) ** vec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9).T).getInt(0)
      val classidAs = toArray(res).zipWithIndex.reduce((a, v) => if (v._1 > a._1) v else a)._2
      println(s"labeled as   : $labeledAs, classified as: $classidAs - $res")
    }
    println("Bye.")
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
    *
    * @param params the given command line parameters
    * @return a model build with those given parameters
    */
  private def modelFromParams(params: DefaultParams): MLModel =
    Conv3ModelBuilder.build(
      imgSize = params.imgSize,
      depth = params.n_channels,
      seed = params.seed,
      eta = params.eta,
      nc1 = params.nc1,
      nc2 = params.nc2,
      nc3 = params.nc3,
      n_dense = params.n_dense,
      parallel = params.parallel)
}