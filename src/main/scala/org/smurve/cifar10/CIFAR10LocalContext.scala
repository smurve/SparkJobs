package org.smurve.cifar10

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.smurve.cifar10.config.{CIFAR10Config, MyHDFSConfig, MyLocalConfig}

/**
  * creates a context for local execution with files read from either hdfs or the local file system
  */
class CIFAR10LocalContext(fsContext: String = "local") extends CIFAR10SparkEnv {

  val config: CIFAR10Config = if (fsContext == "hdfs") new MyHDFSConfig else new MyLocalConfig
  val session: SparkSession = SparkSession.builder().appName("MNist").master("local").getOrCreate()
  val sc: SparkContext = session.sparkContext
}
