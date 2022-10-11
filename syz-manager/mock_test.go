package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

func readSyscall(path string) (map[string]int, []string) {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	content, _ := ioutil.ReadAll(file)
	contentStr := string(content)
	syscallList := strings.Split(contentStr, " ")
	sys2index := make(map[string]int)
	for i := 0; i < len(syscallList); i++ {
		sys2index[syscallList[i]] = i
	}
	return sys2index, syscallList
}

func transfer(calls []string, sys2index map[string]int) []int {
	var calls_int []int
	for _, c := range calls {
		calls_int = append(calls_int, sys2index[c])
	}
	return calls_int
}

func TestMutate(t *testing.T) {
	model := NewModel("/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt", 0)
	sys2index, sysList := readSyscall("/home/syz-mocking/tools/model_manager/api/lang_model/data/targetSyscalls")
	calls := []string{"socket$igmp6"}
	calls_int := transfer(calls, sys2index)

	idx := Mutate(model, calls_int, len(calls_int))
	fmt.Printf("predicted call: %v", sysList[idx])
}
