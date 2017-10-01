
name := "ml.experiments"
version := "1.0"
scalaVersion := "2.11.8"
organization := "org.smurve"
val nd4jVersion = "0.9.1"
val dl4jVersion = "0.9.1"
val cudaversion = "8.0"

val targetPlatform = "linux-x86_64"

ivyConfigurations += config("compileonly").hide

excludeDependencies += "org.slf4j" % "slf4j-log4j12"

assemblyMergeStrategy in assembly := {
  //see https://stackoverflow.com/questions/17265002/hadoop-no-filesystem-for-scheme-file
  case PathList("META-INF", "services", "org.apache.hadoop.fs.FileSystem") => MergeStrategy.filterDistinctLines
  case PathList("META-INF", _@_*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}

val cluster = sys.props.get("cluster").orElse(Some("false")).get

/**
  * to be able to compile the code on any dev machine, the cuda library must be used as "provided". Thus the code compiles but simply
  * doesn't have an impact on performance when used in a non-cuda environment
  */
val localLibs = Seq(
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2"
  //"org.nd4j" % s"nd4j-cuda-$cudaversion-platform" % nd4jVersion % "compileonly" classifier "" classifier targetPlatform
)

val clusterLibs = Seq(
  "org.nd4j" % s"nd4j-cuda-$cudaversion-platform" % nd4jVersion classifier "" classifier targetPlatform
)

libraryDependencies ++= Seq(
  "org.clapper" %% "grizzled-slf4j" % "1.3.1",
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion % "test", //classifier "" classifier "linux-x86_64",
  "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.6", // for visualization
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "org.nd4j" %% "nd4s" % nd4jVersion,
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-nn" % dl4jVersion,
  "org.deeplearning4j" %% "deeplearning4j-ui" % dl4jVersion,
  "org.deeplearning4j" %% "deeplearning4j-parallel-wrapper" % nd4jVersion,
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-streaming" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "org.apache.hadoop" % "hadoop-client" % "2.7.3",
  "com.github.scopt" %% "scopt" % "3.5.0"
)

/**
  * run sbt -Dcluster=true to include the cluster-specific libraries
  */
val specific = if (cluster == "true") clusterLibs else localLibs

libraryDependencies ++= specific

unmanagedClasspath in Compile ++=
  update.value.select(configurationFilter("compileonly"))
