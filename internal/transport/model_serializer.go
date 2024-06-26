package transport

import (
	"encoding/json"
	"fmt"
	"github.com/Vaniog/lirego/internal/ml"
	"github.com/Vaniog/lirego/internal/ml/training"
	"github.com/Vaniog/lirego/internal/transport/generated"
	"github.com/google/uuid"
	"gonum.org/v1/gonum/mat"
	"os"
	"path"
	"reflect"
	"strings"
)

type ModelSerializer struct {
	path string
}

func getModel(tp string, cfg ml.Config, otherParams ...float64) ml.Model {
	switch tp {
	case "LinearModel":
		return ml.NewLinearModel(cfg)
	case "PolynomialModel":
		var deg int
		if len(otherParams) > 0 {
			deg = int(otherParams[0])
		} else {
			deg = 2
		}
		return ml.NewPolynomialModel(cfg, deg)

	}
	return nil
}

func getTrainer(config *generated.TrainerConfig) training.Trainer {
	switch config.Type {
	case "GreedyTrainer":
		return training.NewGreedyTrainer(
			int(config.Params[0]),
			config.Params[1],
		)
	case "BatchTrainer":
		return training.NewBatchTrainer(
			int(config.Params[0]),
			int(config.Params[1]),
			getLearningRate(config.Params[2:]),
		)
	case "GoDeepTrainer":
		return training.NewGoDeepTrainer(
			int(config.Params[0]),
		)
	}

	return nil
}

func getLearningRate(params []float64) training.LearningRate {
	switch int(params[0]) {
	case 0:
		return training.ConstLearningRate(params[1])
	case 1:
		return training.GeometricLearningRate(params[1], params[2])
	case 2:
		return training.ExponentialLearningRate(params[1], params[2])
	}
	return nil
}

func getLoss(tp string) ml.Loss {
	switch tp {
	case "MSELoss":
		return ml.MSELoss{}
	}
	return nil
}

func getRegularizator(tp string) ml.Regularizator {
	switch tp {
	case "EmptyRegularizator":
		return ml.EmptyRegularizator{}
	}
	return nil
}

func getModelName(model ml.Model) string {
	return strings.Replace(reflect.ValueOf(model).Type().String(), "*ml.", "", -1)
}

func NewModelSerializer(path string) *ModelSerializer {
	return &ModelSerializer{
		path: path,
	}
}

func (s *ModelSerializer) Serialize(model ml.Model) (string, error) {
	fmt.Println(model.Bias())
	fmt.Println(model.Weights())
	fmt.Println(model.Weights().RawVector().Data)
	transportModel := &generated.Model{
		Weights:     model.Weights().RawVector().Data,
		Type:        getModelName(model),
		Bias:        model.Bias(),
		RowLen:      int64(model.Config().RowLen),
		OtherParams: getOtherParams(model),
	}
	id := uuid.New()
	serialized, err := json.Marshal(transportModel)
	if err != nil {
		return "", err
	}
	err = s.writeToFile(id.String(), serialized)
	if err != nil {
		return id.String(), err
	}
	return id.String(), nil
}

func getOtherParams(m interface{}) []float64 {
	switch m := m.(type) {
	case *ml.PolynomialModel:
		return []float64{float64(m.Degree())}
	}
	return nil
}

func (s *ModelSerializer) Deserialize(id string) (ml.Model, error) {
	raw, err := s.readFromFile(id)
	if err != nil {
		return nil, err
	}
	var transportModel generated.Model

	err = json.Unmarshal(raw, &transportModel)
	if err != nil {
		return nil, err
	}
	model := getModel(transportModel.Type, ml.Config{RowLen: int(transportModel.RowLen), Bias: true}, transportModel.OtherParams...)
	model.SetWeights(mat.NewVecDense(len(transportModel.Weights), transportModel.Weights))
	model.SetBias(transportModel.Bias)
	return model, nil
}

func (s *ModelSerializer) readFromFile(id string) ([]byte, error) {
	return os.ReadFile(path.Join(s.path, id))
}
func (s *ModelSerializer) writeToFile(id string, data []byte) error {
	err := s.mkdirPath()
	if err != nil {
		return err
	}
	return os.WriteFile(path.Join(s.path, id), data, 0777)
}

func (s *ModelSerializer) mkdirPath() error {
	ex, err := exists(s.path)
	if err == nil && !ex {
		return os.MkdirAll(s.path, os.ModePerm)
	}
	return err
}

func exists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
