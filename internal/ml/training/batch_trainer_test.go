package training

import (
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestBatchTrainer_LinearModel(t *testing.T) {
	ds := NewSliceDataSet([][]float64{
		{0.5, 1},
		{1, 2},
		{2, 4},
		{5, 10},
	})

	trainer := NewBatchTrainer(
		2,
		1000,
		ConstLearningRate(0.1),
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

func TestBatchTrainer_LinearModelWithBias(t *testing.T) {
	ds := NewSliceDataSet([][]float64{
		{0.5, 2},
		{1, 3},
		{2, 5},
		{5, 11},
	})

	trainer := NewBatchTrainer(
		2,
		1000,
		ConstLearningRate(0.1),
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

func FuzzBatchTrainer_LinearModelFuzz(f *testing.F) {
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

		trainer := NewBatchTrainer(
			10,
			1000,
			ConstLearningRate(0.1),
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

func TestBatchTrainer_PolynomialModel(t *testing.T) {
	// y = 2*x*x + 3*x
	ds := NewSliceDataSet([][]float64{
		{0.5, 2},
		{1, 5},
		{2, 14},
		{5, 65},
	})

	trainer := NewBatchTrainer(
		2,
		1000,
		ConstLearningRate(0.01),
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
