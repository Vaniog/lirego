package bencmark

import (
	"fmt"
	"testing"
)

func TestProfiler(t *testing.T) {
	N := 500_000
	f := func() (int, error) {
		res := 0
		arr := make([]int, 0)
		for i := 0; i < N; i++ {
			arr = append(arr, i)
		}
		for _, el := range arr {
			res += el
		}
		return res / N, nil
	}

	_, err, m := Profile(f)
	if err != nil {
		t.Errorf("%e", err)
	}
	fmt.Println(m)
}
