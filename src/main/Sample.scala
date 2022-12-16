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

    // 中心差分による勾配
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
    val steepestDescent: UpdateStrategy =
        (target, weight) =>
            val lerningRate = 0.005
            println("grad: "+Optimizer.grad(target, weight))
            (weight :-: Optimizer.grad(target, weight) :*: lerningRate)

    // 停止条件
    val simpleStopCondition: StopCondition = (target, tmpVal, nextVal) => Optimizer.l1NormGen(Optimizer.grad(target, nextVal)) < 0.0001

// def main(args: Array[String]): Unit = {
//     val target: OptimizableFunction = vec => vec match {case Vector(x, y)=> x*x*x*x-4*x+3*x*x+4*x*x*x+6*y*y case _ => throw new RuntimeException}
//     val firstVal: Weight = Vector(10.0, 0.0)
    
//     // val optimizedWeight = SteepestDescentOptimizer.optimize(target, firstVal, SteepestDescentOptimizer.steepestDescent, SteepestDescentOptimizer.simpleStopCondition)
//     // val optimizedTarget = target(optimizedWeight)

//     // println("Optimized Weight: " + optimizedWeight)
//     // println("Optimized Target Function Value: " + optimizedTarget)
// }

type ActivateFunction = Double => Double
type Value = Vector[Double]
type TeachLabel = Vector[Double]

object Unit:
    def derivative(f: ActivateFunction, point: Double): Double = ???

case class Unit(weight: Weight, activateFunction: ActivateFunction):
    def forward(input: Value): Double =
        activateFunction((input :*: weight).sum)
    def backward(error: Value, state: Double): Double =
        ((error :*: weight).sum) * Unit.derivative(activateFunction, state)

case class Layer(units: Vector[Unit]):
    def forward(input: Value): Value =
        units.map(unit => unit.forward(input))
    def backward(error: Value, states: Value): Value =
        units.zip(states).map((unit, state) => unit.backward(error, state))

case class Network(layers: Vector[Layer]):
    def forward(input: Value, successor: Vector[Layer]): Value =
        if successor.length == 0 then
            return input
        else
            forward(successor.head.forward(input), successor.tail)

    def forwardPropagation(input: Value): Value =
        forward(input, this.layers)

    def backward(error: Value, predecessor: Vector[Layer], predNetsworkStates: Vector[Value]): Value =
        if predecessor.length == 0 then
            return error
        else
            backward(predecessor.head.backward(error, predNetsworkStates.head), predecessor.tail, predNetsworkStates.tail)
    
    def backPropagation(teachLabel: Value, predictedValue: Value, netsworkStates: Vector[Value]): Value =
        backward(predictedValue :-: teachLabel, this.layers.reverse, netsworkStates.reverse)

val af: ActivateFunction = x => 1.0 / (1.0 + math.exp(-x))

val nn = Network(Vector(
    Layer(Vector(
        Unit(Vector(2.0, 3.0), af), Unit(Vector(1.0, -1.0), af), Unit(Vector(1.5, 1.5), af)
    )),
    Layer(Vector(
        Unit(Vector(2.8, 9.0, 7.0), af), Unit(Vector(2.1, -7.0, -2.0), af)
    ))
))

val predictedValue = nn.forwardPropagation(Vector(2.0, 8.0))