package transport

import (
	"fmt"
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestModelSerializer_Serialize(t *testing.T) {
	s := NewModelSerializer("./models")
	id, err := s.Serialize(ml.NewLinearModel(ml.Config{
		RowLen: 10,
		Loss:   nil,
		Reg:    nil,
	}))
	assert.NoError(t, err)
	fmt.Println(id)
}

func TestModelSerializer_Deserialize(t *testing.T) {
	id := "fd3c3abd-4d2a-4cdd-8825-78cc83195029"
	s := NewModelSerializer("./models")
	expected := ml.NewLinearModel(ml.Config{RowLen: 10})
	expected.Weights().CopyVec(mat.NewVecDense(10, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
	model, err := s.Deserialize(id)
	assert.NoError(t, err)
	assert.Equal(t, expected, model)
}
