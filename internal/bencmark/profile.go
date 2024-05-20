package bencmark

import (
	"github.com/Vaniog/lirego/internal/transport/generated"
	"runtime"
	"time"
)

func measureMemory() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return bToKb(m.TotalAlloc)
}

func bToKb(b uint64) uint64 {
	return b / 1024
}

func Profile[T any](f func() (T, error)) (T, error, *generated.Benchmark) {
	baseMemoryUsage := measureMemory()
	startTime := time.Now()

	res, err := f()

	elapsed := time.Since(startTime)
	finalMemoryUsage := measureMemory()
	return res, err, &generated.Benchmark{
		Mem:  int64(finalMemoryUsage - baseMemoryUsage),
		Time: int64(elapsed),
	}
}
