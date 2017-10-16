import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._
import org.smurve.nd4s._
import org.nd4s.Implicits._
val out = vec(-2, 1.1)

val dense = vec(0.6, 0.6, 1.1, 1.1).reshape(2,2)



val inp = vec(0,0,0,1,1,0,1,1).reshape(4,2).T

def activate (v: INDArray) = relu(v - 1)

dense ** inp
val hidden = activate(dense ** inp)
val ybar = activate(out ** hidden)




