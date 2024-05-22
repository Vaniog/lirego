package training

import (
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestGreedyTrainer_LinearModel(t *testing.T) {
	ds := NewSliceDataSet([][]float64{
		{0.5, 1},
		{1, 2},
		{2, 4},
		{5, 10},
	})

	trainer := NewGreedyTrainer(
		100000,
		0.001,
	)

	m := ml.NewLinearModel(ml.Config{
		RowLen: 1,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
		Bias:   false,
	})
	trainer.Train(m, ds)
	assert.True(t, LossScore(m, ds) < 0.1)
}

func TestGreedyTrainer_LinearModelWithBias(t *testing.T) {
	ds := NewSliceDataSet([][]float64{
		{0.5, 2},
		{1, 3},
		{2, 5},
		{5, 11},
	})

	trainer := NewGreedyTrainer(
		100000,
		0.001,
	)

	m := ml.NewLinearModel(ml.Config{
		RowLen: 1,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
		Bias:   true,
	})
	trainer.Train(m, ds)
	assert.True(t, LossScore(m, ds) < 0.1)
}

func FuzzGreedyTrainer_LinearModelFuzz(f *testing.F) {
	f.Fuzz(func(t *testing.T, rowLen int) {
		if rowLen > 20 || rowLen <= 0 {
			return
		}
		ds := genLinearDataSet(
			randFloatSlice(rowLen, 1000),
			100,
		)

		mms := MeanAbsScaler{}
		ds = mms.Transform(ds)

		trainer := NewGreedyTrainer(
			100000,
			0.001,
		)

		m := ml.NewLinearModel(ml.Config{
			RowLen: rowLen,
			Loss:   ml.MSELoss{},
			Reg:    ml.EmptyRegularizator{},
			Bias:   false,
		})
		trainer.Train(m, ds)

		xTrain, yTrain := SplitDataSet(ds)
		yPred := MultiPredict(m, xTrain)

		assert.True(t, R2Score(yPred, yTrain) > 0.8)
	})
}

func TestGreedyTrainer_PolynomialModel(t *testing.T) {
	// y = 2*x*x + 3*x
	ds := NewSliceDataSet([][]float64{
		{0.5, 2},
		{1, 5},
		{2, 14},
		{5, 65},
	})

	trainer := NewGreedyTrainer(
		100000,
		0.01,
	)

	m := ml.NewPolynomialModel(ml.Config{
		RowLen: 1,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
		Bias:   false,
	}, 2)
	trainer.Train(m, ds)
	assert.True(t, LossScore(m, ds) < 0.1)
}

func randFloatSlice(size int, maxAbs float64) []float64 {
	x := make([]float64, size)
	for i := range x {
		x[i] = (rand.Float64() - 0.5) * maxAbs * 2
	}
	return x
}

func genLinearDataSet(coeffs []float64, size int) DataSet {
	data := make([][]float64, 0)

	w := mat.NewVecDense(len(coeffs), coeffs)
	for range size {
		x := randFloatSlice(len(coeffs), 1000)
		xV := mat.NewVecDense(len(x), x)
		data = append(data, slices.Concat(x, []float64{mat.Dot(w, xV)}))
	}
	return NewSliceDataSet(data)
}
