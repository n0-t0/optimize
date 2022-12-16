// ベクトル演算の定義
extension (a: Vector[Double])
  def :+:(b: Vector[Double]): Vector[Double] = a.zip(b).map(_+_)
  def :-:(b: Vector[Double]): Vector[Double] = a.zip(b).map(_-_)
  def :*:(b: Vector[Double]): Vector[Double] = a.zip(b).map(_*_)
  def :/:(b: Vector[Double]): Vector[Double] = a.zip(b).map(_/_)
  def :%:(b: Vector[Double]): Vector[Double] = a.zip(b).map(_%_)
  def :+:(k: Double): Vector[Double] = a.map(_+k)
  def :-:(k: Double): Vector[Double] = a.map(_-k)
  def :*:(k: Double): Vector[Double] = a.map(_*k)
  def :/:(k: Double): Vector[Double] = a.map(_/k)
  def :%:(k: Double): Vector[Double] = a.map(_%k)
  def dot(b: Vector[Double]): Double = (a :*: b).sum

type UnitLinearFlow = Vector[Double] => Double
type UnitNonLinearFlow = Double => Double
type UnitForwardFlow = Vector[Double] => Double
type LayerForwardFlow = Vector[Double] => Vector[Double]
type NetworkForwardFlow = Vector[Double] => Vector[Double]

type UnitBackwardFlow = Vector[Double] => Double
type LayerBackwardFlow = Vector[Double] => Vector[Double]
type NetworkBackwardFlow = Vector[Double] => Vector[Double]



def derivative(f: UnitNonLinearFlow, point: Double, h: Double = 0.0001): Double =
    (f(point+h) - f(point-h))/(2*h)


def makeUnitForwardFlow(f: UnitNonLinearFlow, w: UnitLinearFlow): UnitForwardFlow =
    inputValues =>
        f(w(inputValues))

def makeLayerForwardFlow(f: UnitNonLinearFlow, ws: Vector[UnitLinearFlow]): LayerForwardFlow =
    inputValues =>
        ws.map(makeUnitForwardFlow(f, _)(inputValues))

def makeNetworkForwardFlow(fs: Vector[UnitNonLinearFlow], wss: Vector[Vector[UnitLinearFlow]]): NetworkForwardFlow =
    inputValues =>
        val layerForwardFlows = wss.zip(fs).map((ws, f) => makeLayerForwardFlow(f, ws))
        layerForwardFlows.foldLeft(inputValues)((vec, layerFlow) => layerFlow(vec))


def makeUnitBackwardFlow(inputXs: Vector[Double], f: UnitNonLinearFlow, w: UnitLinearFlow): UnitBackwardFlow =
    inputDeltas =>
        derivative(f, w(inputXs)) * w(inputDeltas)

def makeLayerBackwardFlow(inputXs: Vector[Double], f: UnitNonLinearFlow, ws: Vector[UnitLinearFlow]): LayerBackwardFlow =
    inputDeltas =>
        ws.map(makeUnitBackwardFlow(inputXs, f, _)(inputDeltas))

def makeNetworkBackwardFlow(inputXss: Vector[Vector[Double]], fs: Vector[UnitNonLinearFlow], wss: Vector[Vector[UnitLinearFlow]]): NetworkBackwardFlow =
    inputDeltas =>
        val layerBackwardFlows = wss.zip(fs).zip(inputXss).map{case ((ws, f), inputXs) => makeLayerBackwardFlow(inputXs, f, ws)}
        layerBackwardFlows.foldRight(inputDeltas)((layerFlow, vec) => layerFlow(vec))

def forwardAndBackward(inputValues: Vector[Double], fs: Vector[UnitNonLinearFlow], wss: Vector[Vector[UnitLinearFlow]]): Vector[Vector[Double]] =
    def partialNetworks(wss: Vector[Vector[UnitLinearFlow]], seq: Vector[Vector[Vector[UnitLinearFlow]]]): Vector[Vector[Vector[UnitLinearFlow]]] =
        val wssRev = wss.reverse
        if wssRev.length == 0 then seq else partialNetworks(wssRev.tail, Vector(wssRev.head) +: seq.map(wssRev.head +: _))
    def partialBackNetworks(wss: Vector[Vector[UnitLinearFlow]], seq: Vector[Vector[Vector[UnitLinearFlow]]]): Vector[Vector[Vector[UnitLinearFlow]]] =
        if wss.length == 0 then seq else partialBackNetworks(wss.tail, Vector(wss.head) +: seq.map(wss.head +: _))

    val wssParts =partialNetworks(wss, Vector.empty)
    val wssBackParts = partialBackNetworks(wss, Vector.empty)
    val zs = wssParts.map(wssPart => makeNetworkForwardFlow(fs, wssPart)(inputValues))
    val Deltas = wssBackParts.map(wssBackPart => makeNetworkBackwardFlow(zs, fs, wssBackPart)(makeNetworkForwardFlow(fs, wss)(inputValues)))
    
    // Deltaのheadを落としてダミーをアペンドし、積を取る
    val tailDeltas = Deltas.tail.appended(Vector.empty)
    val delEdelw_ = zs.zip(tailDeltas).map(_ :*: _).dropRight(1)
    val delEdelw = delEdelw_.prepended(Vector.fill(wss.head.length)(0.0))
    
    


    


