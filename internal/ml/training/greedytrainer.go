package training

import (
	"github.com/Vaniog/lirego/internal/ml"
	"gonum.org/v1/gonum/mat"
	"log"
)

type GreedyTrainer struct {
	maxIterations int
	targetLoss    float64
	gradStep      float64
}

func NewGreedyTrainer(
	targetLoss float64,
	maxIterations int,
	gradStep float64,
) *GreedyTrainer {
	return &GreedyTrainer{
		targetLoss:    targetLoss,
		maxIterations: maxIterations,
		gradStep:      gradStep,
	}
}

func (gt GreedyTrainer) Train(m ml.Model, ds DataSet) {
	iterations := 0
	for LossScore(m, ds) > gt.targetLoss && iterations < gt.maxIterations {
		grad, biasGrad := lossGrad(m, ds)
		grad.ScaleVec(-1.0*gt.gradStep, grad)
		wUpdated := mat.VecDenseCopyOf(m.Weights())
		wUpdated.AddVec(grad, wUpdated)
		m.SetWeights(wUpdated)
		m.SetBias(m.Bias() - gt.gradStep*biasGrad)
		iterations++
		if iterations%1000 == 0 {
			log.Println(iterations, LossScore(m, ds))
		}
	}
}
