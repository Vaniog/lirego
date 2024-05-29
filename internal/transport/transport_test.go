package transport

import (
	"context"
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/Vaniog/lirego/internal/transport/generated"
	"github.com/stretchr/testify/assert"
	"os"
	"path"
	"reflect"
	"testing"
)

func TestModelSerializer_SerializeLinearModel(t *testing.T) {
	s := NewModelSerializer("./models")
	lm := ml.NewLinearModel(ml.Config{RowLen: 10})
	id, err := s.Serialize(lm)
	assert.NoError(t, err)

	actual, err := s.Deserialize(id)
	assert.NoError(t, err)
	assert.True(t, reflect.DeepEqual(
		lm.Weights().RawVector().Data,
		actual.Weights().RawVector().Data,
	))

	_ = os.Remove(path.Join("models", id))
}

func TestModelSerializer_SerializePolynomialModel(t *testing.T) {
	s := NewModelSerializer("./models")
	pm := ml.NewPolynomialModel(ml.Config{RowLen: 10}, 10)
	id, err := s.Serialize(pm)
	assert.NoError(t, err)

	actual, err := s.Deserialize(id)
	assert.NoError(t, err)
	assert.True(t, reflect.DeepEqual(
		pm.Weights().RawVector().Data,
		actual.Weights().RawVector().Data,
	))

	_ = os.Remove(path.Join("models", id))
}

func TestMlServer_Train(t *testing.T) {
	server := MlServer{nil}
	r, err := server.Train(context.Background(), &generated.TrainRequest{
		Path: "../../test-data/test.csv",
		TrainerConfig: &generated.TrainerConfig{
			Type:   "GreedyTrainer",
			Params: []float64{100, 0.5},
		},
		ModelConfig: &generated.ModelConfig{
			Type:          "PolynomialModel",
			Regularizator: "EmptyRegularizator",
			Loss:          "MSELoss",
			OtherParams:   []float64{3},
		},
	})

	assert.NoError(t, err)
	assert.NotNil(t, r.ModelId)
}
