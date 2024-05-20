package mlplot

import "math/rand/v2"

func AppendNoise(f XYFunc, noise float64) XYFunc {
	return func(x float64) float64 {
		return f(x) + noise*(rand.Float64()-0.5)
	}
}
