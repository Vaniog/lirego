package training

import (
	"github.com/stretchr/testify/assert"
	"os"
	"testing"
)

func dataSetEqual(ds1, ds2 DataSet) bool {
	if ds1.Len() != ds2.Len() {
		return false
	}

	for i := 0; i < ds1.Len(); i++ {
		row1 := ds1.Row(i)
		row2 := ds2.Row(i)

		if row1.Y != row2.Y {
			return false
		}

		if row1.X.Len() != row2.X.Len() {
			return false
		}

		for j := 0; j < row1.X.Len(); j++ {
			if row1.X.AtVec(j) != row2.X.AtVec(j) {
				return false
			}
		}
	}
	return true
}

func TestNewSliceDatasetFromCSV(t *testing.T) {
	content := `1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0`

	expected := NewSliceDataSet([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	})

	tmpfile, err := os.CreateTemp("", "testdata*.csv")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(content)); err != nil {
		t.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatal(err)
	}

	actual, err := NewSliceDatasetFromCSV(tmpfile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.True(t, dataSetEqual(expected, actual))
}
