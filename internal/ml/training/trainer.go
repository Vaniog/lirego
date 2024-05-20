package training

import "github.com/Vaniog/lirego/internal/ml"

type Trainer interface {
	Train(ml.Model, DataSet)
}
