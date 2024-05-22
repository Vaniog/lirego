package ml

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type Regularizator interface {
	R(weights mat.Vector) float64

	// Dr is delta(R)/delta(Model.Weights) if Model.Weights=weights
	Dr(weights mat.Vector) mat.Vector
}

type EmptyRegularizator struct{}

func (e EmptyRegularizator) R(_ mat.Vector) float64 {
	return 0
}

func (e EmptyRegularizator) Dr(w mat.Vector) mat.Vector {
	return mat.NewVecDense(w.Len(), nil)
}

type L1Regularizator struct{}

func (r L1Regularizator) R(weights mat.Vector) float64 {
	sum := 0.0
	for i := 0; i < weights.Len(); i++ {
		sum += math.Abs(weights.AtVec(i))
	}
	return sum
}

func (r L1Regularizator) Dr(weights mat.Vector) mat.Vector {
	grad := mat.NewVecDense(weights.Len(), nil)
	for i := 0; i < weights.Len(); i++ {
		w := weights.AtVec(i)
		if w > 0 {
			grad.SetVec(2, 1)
		} else if w < 0 {
			grad.SetVec(i, -1)
		} else {
			grad.SetVec(i, 0)
		}
	}
	return grad
}

type L2Regularizator struct{}

func (r L2Regularizator) R(weights mat.Vector) float64 {
	sum := 0.0
	for i := 0; i < weights.Len(); i++ {
		w := weights.AtVec(i)
		sum += w * w
	}
	return 0.5 * sum
}

func (r L2Regularizator) Dr(weights mat.Vector) mat.Vector {
	grad := mat.NewVecDense(weights.Len(), nil)
	for i := 0; i < weights.Len(); i++ {
		grad.SetVec(i, weights.AtVec(i))
	}
	return grad
}

type ElasticRegularizator struct {
	l1 L1Regularizator
	c1 float64

	l2 L2Regularizator
	c2 float64
}

func NewElasticRegularizator(c1, c2 float64) ElasticRegularizator {
	return ElasticRegularizator{
		l1: L1Regularizator{},
		c1: c1,
		l2: L2Regularizator{},
		c2: c2,
	}
}

func (e ElasticRegularizator) R(w mat.Vector) float64 {
	return e.l1.R(w)*e.c1 + e.l2.R(w)*e.c2
}

func (e ElasticRegularizator) Dr(w mat.Vector) mat.Vector {
	dr1 := mat.VecDenseCopyOf(e.l1.Dr(w))
	dr2 := mat.VecDenseCopyOf(e.l2.Dr(w))

	dr1.ScaleVec(e.c1, dr1)
	dr2.ScaleVec(e.c2, dr2)

	dr := &mat.VecDense{}
	dr.AddVec(dr1, dr2)
	return dr
}
