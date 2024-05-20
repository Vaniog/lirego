package ml

type Loss interface {
	F(yPred, y float64) float64

	// Df is delta(F)/delta(Model.Predict) if Model.Predict=yPred, y is const
	Df(yPred, y float64) float64
}

type MSELoss struct {
}

func (s MSELoss) F(yPred, y float64) float64 {
	return 0.5 * (yPred - y) * (yPred - y)
}

func (s MSELoss) Df(yPred, y float64) float64 {
	return yPred - y
}
