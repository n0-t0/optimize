type Weight = Vector[Double]
type OptimizableFunction = Weight => Double
type UpdateStrategy = (OptimizableFunction, Weight) => Weight
type StopCondition = (OptimizableFunction, Weight, Weight) => Boolean

// ベクトル演算の定義
extension (a: Weight)
  def :+:(b: Weight): Weight = a.zip(b).map(_+_)
  def :-:(b: Weight): Weight = a.zip(b).map(_-_)
  def :*:(b: Weight): Weight = a.zip(b).map(_*_)
  def :/:(b: Weight): Weight = a.zip(b).map(_/_)
  def :%:(b: Weight): Weight = a.zip(b).map(_%_)
  def :+:(k: Double): Weight = a.map(_+k)
  def :-:(k: Double): Weight = a.map(_-k)
  def :*:(k: Double): Weight = a.map(_*k)
  def :/:(k: Double): Weight = a.map(_/k)
  def :%:(k: Double): Weight = a.map(_%k)

// 最適化手法がこれを継承する
trait Optimizer:
    def optimize[U <: UpdateStrategy](target: OptimizableFunction, fstVal: Weight, updateStrat: U, stopCond: StopCondition): Weight

// ユーティリティ
object Optimizer:
    val unitVecGen: (Int, Int) => Weight = (index: Int, dim: Int) => {
        Vector.fill(dim)(0).zipWithIndex.map{
            case (value, i) if i == index => 1.0
            case _ => 0.0
        }
    }

    val l1NormGen: Weight => Double = weight =>
        weight.map(scala.math.abs(_)).sum

    val grad: (OptimizableFunction, Weight) => Weight = (target, weight) =>
            val h = 0.000001
            weight.zipWithIndex.map((w_i, i) => 
                (target(weight :+: (unitVecGen(i, weight.length) :*: h ))
                - target(weight :-: (unitVecGen(i, weight.length) :*: h )))
                / (2.0*h)
            )

// 最急降下法
object SteepestDescentOptimizer extends Optimizer:
    // 反復法
    @scala.annotation.tailrec
    def optimize[U <: UpdateStrategy](target: OptimizableFunction, tmpVal: Weight, updateStrategy: U, stopCondition: StopCondition): Weight =
        print("weight: "+tmpVal+",    ")
        val nextVal = updateStrategy(target, tmpVal)
        val continue = stopCondition(target, tmpVal, nextVal)
        if(continue) then
            return nextVal
        else
            optimize(target, nextVal, updateStrategy, stopCondition)

    // 更新規則
    val centralDifference: UpdateStrategy =
        (target, weight) =>
            val lerningRate = 0.005
            println("grad: "+Optimizer.grad(target, weight))
            (weight :-: Optimizer.grad(target, weight) :*: lerningRate)

    // 停止条件
    val simpleStopCondition: StopCondition = (target, tmpVal, nextVal) => Optimizer.l1NormGen(Optimizer.grad(target, nextVal)) < 0.0001

def main(args: Array[String]): Unit = {
    val target: OptimizableFunction = vec => vec match {case Vector(x, y)=> x*x*x*x-4*x+3*x*x+4*x*x*x+6*y*y case _ => throw new RuntimeException}
    val firstVal: Weight = Vector(10.0, 0.0)
    
    val optimizedWeight = SteepestDescentOptimizer.optimize(target, firstVal, SteepestDescentOptimizer.centralDifference, SteepestDescentOptimizer.simpleStopCondition)
    val optimizedTarget = target(optimizedWeight)

    println("Optimized Weight: " + optimizedWeight)
    println("Optimized Target Function Value: " + optimizedTarget)
}

