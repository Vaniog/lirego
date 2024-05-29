package training

import (
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"gonum.org/v1/gonum/mat"
)

type GoDeepTrainer struct {
	maxIterations int
}

func NewGoDeepTrainer(
	maxIterations int,
) *GoDeepTrainer {
	return &GoDeepTrainer{
		maxIterations: maxIterations,
	}
}

func (gt GoDeepTrainer) Train(m ml.Model, ds DataSet) {
	var data []training.Example
	for i := range ds.Len() {
		row := ds.Row(i)
		data = append(data,
			training.Example{
				Input:    mat.VecDenseCopyOf(row.X).RawVector().Data,
				Response: []float64{row.Y},
			})
	}

	neural := deep.NewNeural(&deep.Config{
		Inputs:     ds.Dim(),
		Layout:     []int{1},
		Activation: deep.ActivationNone,
		Mode:       deep.ModeRegression,
		Weight:     deep.NewNormal(1, 0),
		Bias:       true,
	})
	trainer := training.NewBatchTrainer(
		training.NewSGD(0.005, 0.1, 0, false),
		0,
		300,
		16,
	)

	trainer.Train(neural, data, data, gt.maxIterations)

	weights := make([]float64, 0)
	for i := range len(data[0].Input) {
		weights = append(weights, neural.Layers[0].Neurons[0].In[i].Weight)
	}
	m.SetWeights(mat.NewVecDense(len(weights), weights))
	m.SetBias(data[0].Response[0] - neural.Predict(data[0].Input)[0])
}
