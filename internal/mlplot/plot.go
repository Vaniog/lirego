package mlplot

import (
	"errors"
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/Vaniog/lirego/internal/ml/training"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"log"
)

var ErrIllegalPlot = errors.New("can't plot this object")

type XYFunc func(x float64) float64

func PlotFunction(p *plot.Plot, f XYFunc, from, to float64, n int) {
	pts := make(plotter.XYs, n)
	for i := range n {
		x := from + (to-from)*float64(i)/float64(n)
		pts[i].X = x
		pts[i].Y = f(x)
	}
	line, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}
	p.Add(line)
}

func PlotDataSet(p *plot.Plot, ds training.DataSet) {
	p.Title.Text = "DataSet Plot"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	// Create a scatter plotter and fill it with data
	pts := make(plotter.XYs, ds.Len())
	for i := 0; i < ds.Len(); i++ {
		row := ds.Row(i)
		pts[i].X = row.X.AtVec(0)
		pts[i].Y = row.Y
	}

	scatter, _ := plotter.NewScatter(pts)
	p.Add(scatter)
}

func PlotModel(p *plot.Plot, m ml.Model) {
	if m.Config().RowLen != 1 {
		panic(ErrIllegalPlot)
	}
	PlotFunction(p, func(x float64) float64 {
		return m.Predict(mat.NewVecDense(1, []float64{x}))
	}, -1, 1, 100)
}

func PlotSave(p *plot.Plot, path string) {
	if err := p.Save(4*vg.Inch, 4*vg.Inch, path); err != nil {
		log.Fatalf("could not save plot: %v", err)
	}
}

func PredictWithLinearAndPlot(ds training.DataSet) {
	if ds.Dim() != 1 {
		panic(ErrIllegalPlot)
	}

	lm := ml.NewLinearModel(ml.Config{
		RowLen: 1,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
		Bias:   true,
	})

	trainer := training.NewGreedyTrainer(
		10000,
		0.005,
	)

	trainer.Train(lm, ds)

	p := plot.New()
	PlotDataSet(p, ds)
	PlotModel(p, lm)
	PlotSave(p, "dataset.svg")
}

func PredictWithPolynomialAndPlot(ds training.DataSet, degree int, imagePath string) {
	if ds.Dim() != 1 {
		panic(ErrIllegalPlot)
	}

	pm := ml.NewPolynomialModel(ml.Config{
		RowLen: 1,
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
		Bias:   true,
	}, degree)

	trainer := training.NewBatchTrainer(
		50,
		training.ConstLearningRate(1),
		10000,
	)

	trainer.Train(pm, ds)

	p := plot.New()
	PlotDataSet(p, ds)
	PlotModel(p, pm)
	PlotSave(p, imagePath)
}
