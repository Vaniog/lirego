package training

import "math"

type LearningRate func(n int) float64

func ConstLearningRate(c float64) LearningRate {
	return func(_ int) float64 {
		return c
	}
}

func GeometricLearningRate(s, d float64) LearningRate {
	return func(n int) float64 {
		return s * math.Pow(d, float64(n))
	}
}

func ExponentialLearningRate(s, d float64) LearningRate {
	return func(n int) float64 {
		return s * math.Exp(float64(n)*d)
	}
}
