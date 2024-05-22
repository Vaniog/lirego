package ml

func NewLinearModel(c Config) *PolynomialModel {
	return NewPolynomialModel(c, 1)
}
