package training

import (
	"encoding/csv"
	"errors"
	"github.com/Vaniog/lirego/internal/filter"
	"github.com/Vaniog/lirego/internal/ml"
	"gonum.org/v1/gonum/mat"
	"os"
	"strconv"
)

var ErrEmptyFile = errors.New("empty file")
var ErrBadData = errors.New("bad data")

type Row struct {
	X mat.Vector
	Y float64
}

type DataSet interface {
	Row(idx int) Row
	Len() int
	Dim() int
}

func NewRow(row []float64) Row {
	return Row{
		X: mat.NewVecDense(len(row)-1, row[0:len(row)-1]),
		Y: row[len(row)-1],
	}
}

type SliceDataSet struct {
	Rows []Row
}

func NewSliceDataSet(rows [][]float64) *SliceDataSet {
	return &SliceDataSet{
		Rows: filter.Map(rows, NewRow),
	}
}

func (rd *SliceDataSet) Dim() int {
	if len(rd.Rows) == 0 {
		panic("empty dataset")
	}
	return rd.Rows[0].X.Len()
}

func (rd *SliceDataSet) Row(idx int) Row {
	return rd.Rows[idx]
}

func (rd *SliceDataSet) Len() int {
	return len(rd.Rows)
}

func SplitDataSet(ds DataSet) (xs []mat.Vector, ys []float64) {
	for i := range ds.Len() {
		xs = append(xs, ds.Row(i).X)
		ys = append(ys, ds.Row(i).Y)
	}
	return
}

func JoinDataSet(xs []mat.Vector, ys []float64) DataSet {
	var data [][]float64
	for i := range xs {
		var row []float64
		for j := range xs[i].Len() {
			row = append(row, xs[i].AtVec(j))
		}
		row = append(row, ys[i])
		data = append(data, row)
	}
	return NewSliceDataSet(data)
}

func MultiPredict(m ml.Model, inputs []mat.Vector) []float64 {
	return filter.Map(inputs, m.Predict)
}

func ReadCsvFile(filePath string) ([][]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records, nil
}

func parseRow(row []string) (Row, error) {
	train, err := filter.MapWithError(row,
		func(s string) (float64, error) {
			return strconv.ParseFloat(s, 64)
		})
	if err != nil {
		return Row{}, err
	}
	return NewRow(train), nil
}

func NewSliceDatasetFromCSV(path string) (DataSet, error) {
	raw, err := ReadCsvFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		return nil, ErrEmptyFile
	}
	if len(raw[0]) < 2 {
		return nil, ErrBadData
	}

	rows := make([]Row, len(raw))
	for i, row := range raw {
		rows[i], err = parseRow(row)

	}
	return &SliceDataSet{
		Rows: rows,
	}, err
}
