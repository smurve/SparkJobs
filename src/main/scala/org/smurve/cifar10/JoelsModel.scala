package org.smurve.cifar10

import java.io.File

import grizzled.slf4j.Logging
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.{ParamAndGradientIterationListener, ScoreIterationListener}
import org.deeplearning4j.parallelism.ParallelWrapper
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4s.Implicits._
import org.smurve.cifar10.runner.HyperParams
import org.smurve.dl.input.LabeledData
import org.smurve.dl4j.ActivationChecker


class JoelsModel(n_classes: Int = 10, width: Int = 32, height: Int = 32, depth: Int = 3,
                 nf_1: Int = 32, nf_2: Int = 64, nf_3: Int = 128, n_dense: Int = 1024,
                 eta: Double, seed: Int = 5432) extends CIFAR10Tools with Logging{


  lazy val cudaEnv: Boolean = {
    try {
      Class.forName("org.nd4j.jita.conf.CudaEnvironment")
      printf("Found class CudaEnvironment. Trying to access NVIDIA's GPUs")
      true
    } catch {
      case _: ClassNotFoundException =>
        printf("Didn't find class CudaEnvironment. Executing on CPUs")
        false
    }
  }

  /*
    if (cudaEnv) {
      import org.nd4j.jita.conf.CudaEnvironment
      CudaEnvironment.getInstance().getConfiguration
        // key option enabled
        .allowMultiGPU(true)

        // we're allowing larger memory caches
        .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

        // cross-device access is used for faster model averaging over pcie
        .allowCrossDeviceAccess(false)
    }*/


  val model: MultiLayerNetwork = createModel(depth)


  def setupUI(): Unit = {
    System.setProperty("org.deeplearning4j.ui.port", "9090")
    import org.deeplearning4j.ui.api.UIServer
    import org.deeplearning4j.ui.stats.StatsListener
    import org.deeplearning4j.ui.storage.InMemoryStatsStorage
    //Initialize the user interface backend//Initialize the user interface backend

    val uiServer = UIServer.getInstance

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Then add the StatsListener to collect this information from the network, as it trains
    model.setListeners(new StatsListener(statsStorage,1))

  }

  setupUI()

  def this(hyperParams: HyperParams) = this(eta = hyperParams.eta)


  def train(trainingSet: DataSetIterator, testSet: DataSetIterator): Unit = {

    info("Starting training...")

    val reportEvery = 10

    if (cudaEnv) {

      info("Using ParallelWrapper...")
      val pw: ParallelWrapper = new ParallelWrapper.Builder(model)
        // DataSets prefetching options. Set this value with respect to number of actual devices
        .prefetchBuffer(2)

        // set number of workers equal to number of available devices. x1-x2 are good values to start with
        .workers(2)

        // rare averaging improves performance, but might reduce model accuracy
        .averagingFrequency(3)

        // if set to TRUE, on every averaging model score will be reported
        .reportScoreAfterAveraging(true)

        .build()

      pw.fit(trainingSet)

    } else {

      var mini = 1
      while ( trainingSet.hasNext) {
        val nextMinibatch = trainingSet.next()
        println(s"Minibatch No: $mini with ${nextMinibatch.getFeatures.size(0)} images" )
        model.fit(nextMinibatch)
        if ( mini % reportEvery == 0 ) {
          val eval = model.evaluate(testSet)
          println(eval.stats)
        }
        mini += 1
      }
      //model.fit(trainingSet)
    }


    println("Training done. Evaluating...")

    val eval = model.evaluate(testSet)

    println(eval.stats)

  }


  def train(data: LabeledData, n_epochs: Int, n_batches: Int, size_batches: Int): Unit = {

    println("Starting training with Joel's 3-Layer model...")
    model.setListeners(new ScoreIterationListener(1))

    val probe = data.training._1(0, 1, ->, ->, ->).reshape(1, 3, 32, 32)

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

        println(eval.stats)
        testData.reset()
      }
    }
    println("Done.")
  }

  def createModel(depth: Int): MultiLayerNetwork = {


    import Activation._
    import SubsamplingLayer._
    import LossFunctions.LossFunction._

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)

      //.regularization(true).l2(0.0005)
      .learningRate(eta) //.000003
      //.lrPolicyDecayRate(1e-6)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAM)

      .list()

      .layer(0, new ConvolutionLayer.Builder(3, 3)
        .nIn(depth)
        .stride(1, 1)
        .activation(RELU)
        .nOut(nf_1)
        .build())

      .layer(1, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .activation(RELU)
        .nOut(nf_2)
        .build())

      .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(3, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .nOut(nf_3)
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

      //.layer(6, new DropoutLayer.Builder(.3).build())

      .layer(6, new OutputLayer.Builder(NEGATIVELOGLIKELIHOOD)
        .nOut(n_classes)
        .activation(SOFTMAX)
        .build())

      .setInputType(InputType.convolutionalFlat(height, width, depth))
      .backprop(true).pretrain(false)


      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

}
