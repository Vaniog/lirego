package transport

import (
	"github.com/Vaniog/lirego/internal/ml"
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
