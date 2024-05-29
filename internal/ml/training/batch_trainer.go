package training

import (
	"github.com/Vaniog/lirego/internal/ml"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
)

type BatchTrainer struct {
	batchSize    int
	iterations   int
	learningRate LearningRate
}

func NewBatchTrainer(
	batchSize int,
	iterations int,
	learningRate LearningRate,
) *BatchTrainer {
	return &BatchTrainer{
		batchSize:    batchSize,
		iterations:   iterations,
		learningRate: learningRate,
	}
}

func (b BatchTrainer) Train(m ml.Model, ds DataSet) {
	for i := range b.iterations {
		b.eraTrain(m, ds, i)
		if i%100 == 0 {
			log.Println(i, LossScore(m, ds))
		}
	}
}

type fullGrad struct {
	wGrad    mat.Vector
	biasGrad float64
}

func (b BatchTrainer) eraTrain(m ml.Model, ds DataSet, iterations int) {
	batches := batchSplit(ds, b.batchSize)
	batchesAmount := len(batches)

	gradChan := make(chan fullGrad)
	defer close(gradChan)

	for i := range batches {
		if len(batches[i].Rows) == 0 {
			batches[i].Rows = append(batches[i].Rows, ds.Row(0))
		}
		go b.batchGrad(m, &batches[i], iterations, gradChan)
	}

	gradSum := mat.NewVecDense(m.Weights().Len(), nil)
	biasGradSum := 0.0
	n := 0
	for g := range gradChan {
		gradSum.AddVec(g.wGrad, gradSum)
		biasGradSum += g.biasGrad
		n++
		if n == batchesAmount {
			break
		}
	}

	biasGrad := biasGradSum / float64(batchesAmount)
	grad := mat.NewVecDense(m.Weights().Len(), nil)
	grad.ScaleVec(-1.0/float64(batchesAmount), gradSum)

	weights := m.Weights()
	weights.AddVec(grad, weights)
	m.SetWeights(weights)
	m.SetBias(m.Bias() - biasGrad)
}

func batchSplit(ds DataSet, batchSize int) []SliceDataSet {
	mark := make([]int, ds.Len())
	batchesAmount := ds.Len() / batchSize
	for i := range mark {
		mark[i] = rand.Intn(batchesAmount)
	}

	batches := make([]SliceDataSet, 0, batchesAmount)
	for range batchesAmount {
		batches = append(batches, SliceDataSet{Rows: make([]Row, 0, batchSize)})
	}
	for i := range mark {
		batches[mark[i]].Rows = append(batches[mark[i]].Rows, ds.Row(i))
	}

	// if empty, add something
	for i := range batches {
		if len(batches[i].Rows) == 0 {
			batches[i].Rows = append(batches[i].Rows, ds.Row(0))
		}
	}

	return batches
}

func (b BatchTrainer) batchGrad(m ml.Model, batch DataSet, iterations int, gradChan chan<- fullGrad) {
	grad, biasGrad := lossGrad(m, batch)
	learningRate := b.learningRate(iterations)
	grad.ScaleVec(learningRate, grad)
	gradChan <- fullGrad{
		wGrad:    grad,
		biasGrad: learningRate * biasGrad,
	}
}
