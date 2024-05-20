package training

import "math"

func R2Score(yTrue, yPred []float64) float64 {
	meanY := mean(yTrue)

	var ssTot, ssRes float64
	for i := 0; i < len(yTrue); i++ {
		ssTot += math.Pow(yTrue[i]-meanY, 2)
		ssRes += math.Pow(yTrue[i]-yPred[i], 2)
	}

	r2 := 1 - (ssRes / ssTot)
	return r2
}

func mean(data []float64) float64 {
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data))
}
