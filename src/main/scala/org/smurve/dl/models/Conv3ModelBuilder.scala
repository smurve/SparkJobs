package org.smurve.dl.models

import grizzled.slf4j.Logging
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.parallelism.ParallelWrapper
import org.nd4j.linalg.activations.Activation._
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction._
import org.smurve.dl.dl4j.{MLModel, MultiLayerAdapter, ParallelWrapperAdapter}

/**
  * creates a 3-layer convolutional network
  */
object Conv3ModelBuilder extends Logging {


  def build(seed: Int,
            eta: Double = 1e-0,
            n_classes: Int = 10,
            imgSize: Int = 32,
            depth: Int = 3,
            nc1: Int = 10,
            nc2: Int = 10,
            nc3: Int = 10,
            n_dense: Int = 100,
            updater: Updater = Updater.ADAM,
            parallel: Int = 1
           ): MLModel = {

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .learningRate(eta)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(updater)

      .list()

      .layer(0, new ConvolutionLayer.Builder(3, 3)
        .nIn(depth)
        .stride(1, 1)
        .activation(RELU)
        .nOut(nc1)
        .build())

      .layer(1, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .activation(RELU)
        .nOut(nc2)
        .build())

      .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(3, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .nOut(nc3)
        .activation(RELU)
        .build())

      .layer(4, new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(5, new DenseLayer.Builder()
        .activation(RELU)
        .nOut(n_dense)
        .build())

      .layer(6, new DropoutLayer.Builder(.3).build())

      .layer(6, new OutputLayer.Builder(NEGATIVELOGLIKELIHOOD)
        .nOut(n_classes)
        .activation(SOFTMAX)
        .build())

      .setInputType(InputType.convolutionalFlat(imgSize, imgSize, depth))
      .backprop(true).pretrain(false)

      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    if ( parallel > 1 ) {
      info(s"$parallel workers required, using parallel wrapper...")
      new ParallelWrapperAdapter(new ParallelWrapper.Builder(model)
        .workers(parallel)
        .averagingFrequency(1)
        .reportScoreAfterAveraging(true)
        //  .trainingMode(TrainingMode.SHARED_GRADIENTS)
        .build(), model)
    }
    else {
      info(s"Single worker required, using trivial adapter...")
      new MultiLayerAdapter(model)
    }

  }

}
