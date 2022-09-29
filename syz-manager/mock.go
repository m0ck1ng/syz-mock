package main

/*
#cgo LDFLAGS: -L./lib -lmock
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "./lib/mock.h"
*/
import "C"
import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"time"
	"unsafe"

	"github.com/google/syzkaller/prog"
)

type Model_t = C.model_t

func DefaultModel() *Model_t {
	model := C.model_default()
	return model
}

func NewModel(path string, device_id uint) *Model_t {
	path_c := C.CString(path)
	defer C.free(unsafe.Pointer(path_c))
	model := C.model_new(path_c, C.uint(device_id))
	return model
}

func LoadModel(model *Model_t, path string, device_id uint) {
	path_c := C.CString(path)
	defer C.free(unsafe.Pointer(path_c))
	C.model_load(model, path_c, C.uint(device_id))
}

func Generate(model *Model_t, calls []int) int {
	calls_ptr := (*C.int)(unsafe.Pointer(&calls[0]))
	return (int)(C.model_gen(model, calls_ptr, C.uint(len(calls))))
}

func Mutate(model *Model_t, calls []int, idx int) int {
	calls_ptr := (*C.int)(unsafe.Pointer(&calls[0]))
	return (int)(C.model_mutate(model, calls_ptr, C.uint(len(calls)), C.int(idx)))
}

func ModelExists(model *Model_t) bool {
	ret := (uint)(C.model_exists(model))
	return (ret == 1)
}

func (mgr *Manager) newModel(path string, device_id uint) {
	mgr.relModel = NewModel(path, device_id)
}

func (mgr *Manager) defaultModel() {
	mgr.relModel = DefaultModel()
}

func (mgr *Manager) loadModel(path string, device_id uint) {
	LoadModel(mgr.relModel, path, device_id)
}

func (mgr *Manager) checkModel() bool {
	return ModelExists(mgr.relModel)
}

func (mgr *Manager) mutateWithModel(calls []int, idx int) int {
	return Mutate(mgr.relModel, calls, idx)
}

func trainModel(syzdir, workdir string) string {
	client := &http.Client{
		Timeout: time.Minute * 10,
	}
	resp, err := client.PostForm("http://127.0.0.1:8000/api/model",
		url.Values{"syzdir": {syzdir}, "workdir": {workdir}})

	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return string(body)
}

func dumpSyscalls(targetSyscalls []*prog.Syscall) {
	filePath := "/home/workdir/targetSyscalls"
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		fmt.Println("Fail to open file", err)
	}
	defer file.Close()
	write := bufio.NewWriter(file)
	for _, syscall := range targetSyscalls {
		write.WriteString(fmt.Sprintf("%v ", syscall.Name))
	}
	write.Flush()
}

// func main() {
// 	// model := NewModel("/home/workdir/syscall_model_jit_best.pt", 0)
// 	model := DefaultModel()
// 	LoadModel(model, "/home/workdir/syscall_model_jit_best.pt", 1)
// 	exists := ModelExists(model)
// 	fmt.Printf("model exists: %v\n", exists)
// 	call_idx := Generate(model, []uint{4, 5, 6})
// 	fmt.Printf("call id: %v\n", call_idx)
// }
