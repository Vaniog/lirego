package ml

import (
	"gonum.org/v1/gonum/mat"
)

type QuadraticModel struct {
	c    Config
	bias float64
	w0   *mat.VecDense
	w1   *mat.VecDense
	w2   *mat.Dense
}

func NewQuadraticModel(c Config) *QuadraticModel {
	return &QuadraticModel{
		c:  c,
		w0: mat.NewVecDense(c.RowLen, nil),
		w1: mat.NewVecDense(c.RowLen, nil),
		w2: mat.NewDense(c.RowLen, c.RowLen, nil),
	}
}

func (qm *QuadraticModel) Predict(x mat.Vector) float64 {
	yp := mat.Dot(x, qm.w0) // Linear term
	quadraticTerm := mat.NewVecDense(x.Len(), nil)
	quadraticTerm.MulVec(qm.w2, x)
	yp += 0.5 * mat.Dot(x, quadraticTerm) // Quadratic term

	if qm.c.Bias {
		yp += qm.bias
	}
	return yp
}

func (qm *QuadraticModel) Config() Config {
	return qm.c
}

func (qm *QuadraticModel) Weights() *mat.VecDense {
	w := concatVectors(qm.w0, qm.w1)
	flattenedW2 := flattenMatrix(qm.w2)
	w = concatVectors(w, flattenedW2)
	return mat.VecDenseCopyOf(w)
}

func (qm *QuadraticModel) SetWeights(w mat.Vector) {
	rowLen := qm.c.RowLen
	for i := 0; i < rowLen; i++ {
		qm.w0.SetVec(i, w.AtVec(i))
		qm.w1.SetVec(i, w.AtVec(rowLen+i))
	}
	for i := 0; i < rowLen; i++ {
		for j := 0; j < rowLen; j++ {
			qm.w2.Set(i, j, w.AtVec(2*rowLen+i*rowLen+j))
		}
	}
}

func (qm *QuadraticModel) Bias() float64 {
	return qm.bias
}

func (qm *QuadraticModel) SetBias(b float64) {
	qm.bias = b
}

func (qm *QuadraticModel) Dp(x mat.Vector) mat.Vector {
	rowLen := x.Len()
	gradient := mat.NewVecDense(2*rowLen+rowLen*rowLen, nil)

	for i := 0; i < rowLen; i++ {
		gradient.SetVec(i, x.AtVec(i))
	}

	offset := rowLen
	for i := 0; i < rowLen; i++ {
		gradient.SetVec(offset+i, x.AtVec(i))
	}

	offset = 2 * rowLen
	for i := 0; i < rowLen; i++ {
		for j := 0; j < rowLen; j++ {
			gradient.SetVec(offset+i*rowLen+j, x.AtVec(i)*x.AtVec(j))
		}
	}

	return gradient
}

func flattenMatrix(m *mat.Dense) *mat.VecDense {
	rows, cols := m.Dims()
	flattened := mat.NewVecDense(rows*cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flattened.SetVec(i*cols+j, m.At(i, j))
		}
	}
	return flattened
}
