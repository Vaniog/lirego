package ml

import "gonum.org/v1/gonum/mat"

type Regularizator interface {
	R(weights mat.Vector) float64

	// Dr is delta(R)/delta(Model.Weights) if Model.Weights=weights
	Dr(weights mat.Vector) mat.Vector
}

type EmptyRegularizator struct {
}

func (e EmptyRegularizator) R(_ mat.Vector) float64 {
	return 0
}

func (e EmptyRegularizator) Dr(w mat.Vector) mat.Vector {
	return mat.NewVecDense(w.Len(), nil)
}
