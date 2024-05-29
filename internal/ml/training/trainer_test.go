package training

import (
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/stretchr/testify/assert"
	"math/rand"
	"testing"
)

func TestGreedyTrainer(t *testing.T) {
	trainer := NewGreedyTrainer(1000, 0.1)
	testTrainerOnLinear(t, trainer)
	testTrainerOnPolynomial(t, trainer)
}

func TestBatchTrainer(t *testing.T) {
	trainer := NewBatchTrainer(2, 1000, GeometricLearningRate(1, 0.99999))
	testTrainerOnLinear(t, trainer)
	testTrainerOnPolynomial(t, trainer)
}

func TestGoDeepTrainer(t *testing.T) {
	testTrainerOnLinear(t, NewGoDeepTrainer(1000))
}

func testTrainerOnLinear(t *testing.T, trainer Trainer) {
	dsLin1 := datasetFromFunction(1, func(x ...float64) float64 {
		return x[0]*2 + 1
	}, -1, 1, 10)
	lm1 := defaultPolynomial(1, 1)
	trainer.Train(lm1, dsLin1)
	testScore(t, dsLin1, lm1)

	dsLin2 := datasetFromFunction(2, func(x ...float64) float64 {
		return x[0]*2 + x[1]*-3
	}, -1, 1, 20)
	lm2 := defaultPolynomial(2, 1)
	trainer.Train(lm2, dsLin2)
	testScore(t, dsLin2, lm2)
}

func testTrainerOnPolynomial(t *testing.T, trainer Trainer) {
	dsPol1 := datasetFromFunction(2, func(x ...float64) float64 {
		return x[0]*x[0]*-3 + x[1]*2 + 1
	}, -1, 1, 10)
	pm1 := defaultPolynomial(2, 2)
	trainer.Train(pm1, dsPol1)
	testScore(t, dsPol1, pm1)
	dsPol2 := datasetFromFunction(2, func(x ...float64) float64 {
		return x[0]*x[0]*-2 + x[1]*x[1]*-2 + 1
	}, -1, 1, 10)
	pm2 := defaultPolynomial(2, 3)
	trainer.Train(pm2, dsPol2)
	testScore(t, dsPol2, pm2)
}

func defaultPolynomial(dim, degree int) ml.Model {
	return ml.NewPolynomialModel(ml.Config{
		RowLen: dim,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
		Bias:   true,
	}, degree)
}

func testScore(t *testing.T, ds DataSet, m ml.Model) {
	xs, yTrue := SplitDataSet(ds)
	assert.True(t, R2Score(MultiPredict(m, xs), yTrue) > 0.9)
}

func datasetFromFunction(dim int, f func(x ...float64) float64, from, to float64, n int) DataSet {
	var rows [][]float64
	for range n {
		row := make([]float64, dim)
		for i := range dim {
			row[i] = from + rand.Float64()*(to-from)
		}
		row = append(row, f(row...))
		rows = append(rows, row)
	}

	return NewSliceDataSet(rows)
}
