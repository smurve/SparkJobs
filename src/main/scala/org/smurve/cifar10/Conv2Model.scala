package org.smurve.cifar10

import java.io.File

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.{ParamAndGradientIterationListener, ScoreIterationListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4s.Implicits._
import org.smurve.dl.input.LabeledData
import org.smurve.dl4j.ActivationChecker


class Conv2Model(n_classes: Int = 10, width: Int = 32, height: Int = 32, depth: Int = 3,
                 n_features_1: Int, n_features_2: Int, n_dense: Int, eta: Double, seed: Int = 5432) extends CIFAR10Tools {

  val model: MultiLayerNetwork = createModel( depth )



  def train(data: LabeledData, n_epochs: Int, n_batches: Int, size_batches: Int): Unit = {

    val pgil = new ParamAndGradientIterationListener(
      /*iterations=*/ size_batches,
      /* printHeader = */ true,
      /*printMean=*/ false,
      /* printMinMax=*/ true,
      /*printMeanAbsValue=*/ false,
      /*outputToConsole=*/ false,
      /*outputToFile = */ true,
      /*outputToLogger = */ false,
      /*file = */ new File("stats.csv"),
      /*delimiter = */ ",")

    println("Starting training with a 2-layer convnet model...")
    model.setListeners(new ScoreIterationListener(1), pgil)

    val probe = data.training._1(0, 1, ->, ->, ->).reshape(1,3,32,32)
    val checker = new ActivationChecker(probe, n_channels = depth, imgSize = height)

    /*
    println(probe)
    asImage(probe).output("probe.png")
    */


    for (epoch <- 0 until n_epochs) {
      println(s"Starting epoch Nr. $epoch")
      for (batch <- 0 until n_batches) {
        println(s"Starting batch Nr. $batch")

        val trbs = data.training._1(batch, ->, ->, ->, ->)
        //Nd4j.shuffle(trbs, 1,2,3)
        val trbl = data.training._2(batch, ->, ->)

        val tebs = data.test._1(batch, ->, ->, ->, ->)
        val tebl = data.test._2(batch, ->, ->)

        val trainingData = new ExistingDataSetIterator(new DataSet(trbs, trbl))
        val testData = new ExistingDataSetIterator(new DataSet(tebs, tebl))

        model.fit(trainingData)
        val eval = model.evaluate(testData)

        checker.analyseOutput(model, 3)

        println(eval.stats)
        testData.reset()
      }
    }
    println("Done.")
  }

  def createModel( depth: Int): MultiLayerNetwork = {


    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      //.regularization(true).l2(0.0005)
      .learningRate(eta) //.000003
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAM)

      .list()

      .layer(0, new ConvolutionLayer.Builder(3, 3)
        .nIn(depth)
        .stride(1, 1)
        .activation(Activation.RELU)
        .nOut(n_features_1)
        .build())

      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(2, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .nOut(n_features_2)
        .activation(Activation.RELU)
        .build())

      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(4, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(n_dense)
        .build())

      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(n_classes)
        .activation(Activation.SOFTMAX)
        .build())

      .setInputType(InputType.convolutionalFlat(height, width, depth))
      .backprop(true).pretrain(false)

      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

}
