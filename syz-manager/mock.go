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
	"unsafe"
)

type Model_t = C.model_t

func DefaultModel() *Model_t {
	model := C.model_default()
	return model
}

func NewModel(path string, device_id uint32) *Model_t {
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

func Generate(model *Model_t, calls []uint) int {
	calls_ptr := (*C.uint)(unsafe.Pointer(&calls[0]))
	return (int)(C.model_gen(model, calls_ptr, C.uint(len(calls))))
}

func Mutate(model *Model_t, calls []uint, idx uint) int {
	calls_ptr := (*C.uint)(unsafe.Pointer(&calls[0]))
	return (int)(C.model_mutate(model, calls_ptr, C.uint(len(calls)), C.uint(idx)))
}

func ModelExists(model *Model_t) uint {
	return (uint)(C.model_exists(model))
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
