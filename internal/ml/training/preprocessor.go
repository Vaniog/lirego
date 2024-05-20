package training

import (
	"github.com/Vaniog/lirego/internal/filter"
	"gonum.org/v1/gonum/mat"
	"math"
)

type Preprocessor interface {
	Transform(DataSet) DataSet
}

type MeanAbsScaler struct {
}

func (mms *MeanAbsScaler) Transform(ds DataSet) DataSet {
	dim := ds.Row(0).X.Len()
	xs, ys := SplitDataSet(ds)

	n := min(512, len(xs))
	xMean := mat.NewVecDense(dim, nil)

	for i := range n {
		for j := range xMean.Len() {
			xMean.SetVec(j, xMean.AtVec(j)+math.Abs(xs[i].AtVec(j)))
		}
	}

	xMean.ScaleVec(1.0/float64(n), xMean)

	xs = filter.Map(xs, func(x mat.Vector) mat.Vector {
		v := mat.VecDenseCopyOf(x)
		for i := range xMean.Len() {
			v.SetVec(i, v.AtVec(i)/xMean.AtVec(i))
		}
		return v
	})

	return JoinDataSet(xs, ys)
}
