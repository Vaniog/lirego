package ml

import (
	"gonum.org/v1/gonum/mat"
)

type LinearModel struct {
	w    *mat.VecDense
	bias float64
	c    Config
}

func (lm *LinearModel) Config() Config {
	return lm.c
}

func NewLinearModel(c Config) *LinearModel {
	return &LinearModel{
		w: mat.NewVecDense(c.RowLen, nil),
		c: c,
	}
}

func (lm *LinearModel) Dp(x mat.Vector) mat.Vector {
	return x
}

func (lm *LinearModel) Predict(x mat.Vector) float64 {
	prediction := mat.Dot(lm.w, x)
	if lm.c.Bias {
		prediction += lm.bias
	}
	return prediction
}

func (lm *LinearModel) Weights() *mat.VecDense {
	return mat.VecDenseCopyOf(lm.w)
}
func (lm *LinearModel) SetWeights(v mat.Vector) {
	lm.w = mat.VecDenseCopyOf(v)
}

func (lm *LinearModel) Bias() float64 {
	return lm.bias
}

func (lm *LinearModel) SetBias(b float64) {
	lm.bias = b
}
