package ml

import (
	"github.com/Vaniog/lirego/internal/filter"
	"gonum.org/v1/gonum/mat"
)

type PolynomialModel struct {
	c    Config
	bias float64
	ws   []*mat.VecDense
}

func (pm *PolynomialModel) SetWeights(w mat.Vector) {
	rowLen := pm.Config().RowLen
	for i := range pm.ws {
		for j := range rowLen {
			pm.ws[i].SetVec(j, w.AtVec(i*rowLen+j))
		}
	}
}

func NewPolynomialModel(c Config, degree int) *PolynomialModel {
	pm := PolynomialModel{c: c}
	for range degree {
		pm.ws = append(pm.ws, mat.NewVecDense(c.RowLen, nil))
	}
	return &pm
}

func (pm *PolynomialModel) Predict(x mat.Vector) float64 {
	yp := 0.0
	curX := mat.VecDenseCopyOf(x)
	for i := range pm.ws {
		yp += mat.Dot(curX, pm.ws[i])
		for j := range x.Len() {
			curX.SetVec(j, curX.AtVec(j)*x.AtVec(j))
		}
	}
	if pm.c.Bias {
		yp += pm.bias
	}
	return yp
}

func (pm *PolynomialModel) Config() Config {
	return pm.c
}

func (pm *PolynomialModel) Weights() *mat.VecDense {
	return filter.Reduce(
		pm.ws[1:len(pm.ws)],
		pm.ws[0],
		func(v1, v2 *mat.VecDense) *mat.VecDense {
			return mat.VecDenseCopyOf(concatVectors(v1, v2))
		},
	)
}

func concatVectors(v1, v2 mat.Vector) mat.Vector {
	lenV1 := v1.Len()
	lenV2 := v2.Len()

	result := mat.NewVecDense(lenV1+lenV2, nil)

	for i := 0; i < lenV1; i++ {
		result.SetVec(i, v1.AtVec(i))
	}

	for i := 0; i < lenV2; i++ {
		result.SetVec(lenV1+i, v2.AtVec(i))
	}

	return result
}

func (pm *PolynomialModel) Bias() float64 {
	return pm.bias
}

func (pm *PolynomialModel) SetBias(b float64) {
	pm.bias = b
}

func (pm *PolynomialModel) Dp(x mat.Vector) mat.Vector {
	dp := x
	curX := mat.VecDenseCopyOf(x)
	for i := 1; i < len(pm.ws); i++ {
		for j := range x.Len() {
			curX.SetVec(j, curX.AtVec(j)*x.AtVec(j))
		}
		dp = concatVectors(dp, curX)
	}
	return dp
}
