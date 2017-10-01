package org.smurve.cifar10.runner

import scopt.OptionParser


class CmdLineTool {

  val parser: OptionParser[HyperParams] = new OptionParser[HyperParams]("CIFAR10Runner") {
    head("CIFAR10Runner", "1.0")

    opt[Int]('E', "num-epochs").valueName("Number of Epochs")
      .action((x, args) => args.copy(numEpochs = x))

    opt[Int]('T', "num-training").valueName("Number of Training Images")
      .action((x, args) => args.copy(numTraining = x))

    opt[Int]('t', "num-test").valueName("Number of test Images")
      .action((x, args) => args.copy(numTest = x))

    opt[Int]('b', "batch-size").valueName("Mini Batch size")
      .action((x, args) => args.copy(minibatchSize = x))

    opt[Double]('e', "eta").valueName("Learning rate")
      .action((x, args) => args.copy(eta = x))

    opt[String]('p', "precision").valueName("numerical precision, one of 'd', 'f', 'h'")
      .action((x, args) => args.copy(precision = x))
  }


}
