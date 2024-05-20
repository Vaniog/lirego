package ml

import (
	"gonum.org/v1/gonum/mat"
)

type Model interface {
	Predict(x mat.Vector) float64
	Config() Config
	Weights() *mat.VecDense
	SetWeights(mat.Vector)
	Bias() float64
	SetBias(float64)

	// Dp is delta(Predict)/delta(Weights) if X is const
	Dp(x mat.Vector) mat.Vector
}

type Config struct {
	// RowLen is inputs len
	RowLen int
	Loss   Loss
	Reg    Regularizator
	Bias   bool
}
