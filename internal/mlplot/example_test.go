package mlplot

import (
	"github.com/Vaniog/lirego/internal/ml/training"
	"math"
)

func ExamplePredictWithLinearAndPlot() {
	PredictWithLinearAndPlot(training.NewSliceDataSet([][]float64{
		{1, 2},
		{2, 4},
		{3, 6},
		{4, 8},
		{5, 10},
	}))
	// Output:
}

func ExamplePredictWithPolynomialAndPlot() {
	PredictWithPolynomialAndPlot(DatasetFromFunction(
		AppendNoise(func(x float64) float64 {
			return math.Sin(x*2*math.Pi) / 2
		}, 0.2), -1, 1, 100), 10, "dataset.png")
	// Output:
}
