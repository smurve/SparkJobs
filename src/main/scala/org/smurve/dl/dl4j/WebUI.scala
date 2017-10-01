package org.smurve.dl.dl4j

import grizzled.slf4j.Logging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage

/**
  * Spawn a play server to display relevant statistics during the training
  * @param model the model to inspect
  * @param port the port to provide the web UI at
  * @param frequency the number of iterations before a new reading is provided
  */
class WebUI(model: MultiLayerNetwork, port: Int = 9090, frequency: Int = 1) extends Logging {

  System.setProperty("org.deeplearning4j.ui.port", s"$port")
  val uiServer: UIServer = UIServer.getInstance

  val statsStorage = new InMemoryStatsStorage

  uiServer.attach(statsStorage)

  model.setListeners(new StatsListener(statsStorage, 1))

  sys.ShutdownHookThread {
    info("Shutting down the UI Server...")
    try {
      uiServer.stop()
    } catch {
      case e: Exception =>
        warn(s"caught: $e. Ignoring")
    }
    info("UI Server down.")
  }

}
