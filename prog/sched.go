package prog

import (
	"math"
	"sync/atomic"

	wr "github.com/mroth/weightedrand"
)

const ARMSIZE = 2

type ARM int

const (
	INSERT ARM = iota
	INSERTMOCK
)

// 0 refer to original insert, 1 refer to insert with model
type Sched struct {
	ExecTotal   [ARMSIZE]uint32
	IntTotal    [ARMSIZE]uint32
	weights     [ARMSIZE]uint32
	modelExists atomic.Bool
}

func (sch *Sched) IncExecTotal(pos uint) {
	atomic.AddUint32(&sch.ExecTotal[pos], 1)
}

func (sch *Sched) IncIntTotal(pos uint) {
	atomic.AddUint32(&sch.IntTotal[pos], 1)
}

func (sch *Sched) Reset() {
	for i := 0; i < ARMSIZE; i++ {
		atomic.StoreUint32(&sch.ExecTotal[i], 0)
		atomic.StoreUint32(&sch.IntTotal[i], 0)
		atomic.StoreUint32(&sch.weights[i], 5)
	}
	atomic.StoreUint32(&sch.weights[INSERT], 10)
}

func (sch *Sched) UpdateWeights() {
	maxVal, maxIndex := 0.0, 0
	for i := 0; i < ARMSIZE; i++ {
		ExecTotal := float64(atomic.LoadUint32(&sch.ExecTotal[i]))
		IntTotal := float64(atomic.LoadUint32(&sch.IntTotal[i]))
		item1 := 100.0 * IntTotal / ExecTotal
		item2 := 100.0 * (math.Sqrt(2.0 * math.Log(ExecTotal+1.0) / (IntTotal + 1.0)))
		if item1+item2 > maxVal {
			maxVal = item1 + item2
			maxIndex = i
		}
	}
	atomic.StoreUint32(&sch.weights[INSERT], uint32(1-maxIndex))
	atomic.StoreUint32(&sch.weights[INSERTMOCK], uint32(maxIndex))
}

func (sch *Sched) Choice() int {
	var result int
	if sch.modelExists.Load() {
		chooser, _ := wr.NewChooser(
			wr.Choice{Item: 0, Weight: uint(atomic.LoadUint32(&sch.weights[0]))},
			wr.Choice{Item: 1, Weight: uint(atomic.LoadUint32(&sch.weights[1]))},
		)
		result = chooser.Pick().(int)
	} else {
		result = int(INSERT)
	}
	return result
}

func (sch *Sched) SetModelFlag(flag bool) {
	sch.modelExists.Store(flag)
}

func (sch *Sched) GetModelFlag() bool {
	return sch.modelExists.Load()
}
