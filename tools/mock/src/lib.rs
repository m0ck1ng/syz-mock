extern crate libc;

use tch::{CModule, Tensor, Device};
use libc::{c_char, size_t};
use rand::prelude::*;
use std::{
    ffi::CStr,
    slice,
    collections::HashMap,
};

pub type RngType = SmallRng;
pub type SyscallId = usize;

#[derive(Default, Debug)]
pub struct Model {
    model: Option<CModule>,
    device: Option<Device>,
}

const NUM_LAYERS:i64=2;
const HIDDEN_DIM:i64=128;

impl Model {
    pub fn new<P: AsRef<std::path::Path>>(model_path: P, device_id: usize) -> Self {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(device_id);
        }
        Self {
            model: Some(tch::CModule::load_on_device(model_path, device).unwrap()),
            device: Some(device),
        }
    }

    pub fn load<P: AsRef<std::path::Path>>(&mut self, model_path: P, device_id: usize) {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(device_id);
        }
        self.device = Some(device);
        self.model = Some(tch::CModule::load_on_device(model_path, device).unwrap());
    }

    pub fn eval(&self, inputs: &[usize]) -> Option<Tensor> {
        if self.model.is_none() {
            None
        }
        else {
            // begin
            let length_ival = tch::IValue::Tensor(Tensor::of_slice(&[inputs.len() as u8]).to_device(self.device.unwrap()));
            let inputs = inputs.into_iter().map(|i| *i as i32).collect::<Vec<i32>>();
            let inputs_ival = tch::IValue::Tensor(Tensor::stack(&[Tensor::of_slice(&inputs)], 0).to_device(self.device.unwrap()));
            let h0 = tch::IValue::Tensor(Tensor::zeros(&[NUM_LAYERS*2, 1, HIDDEN_DIM],(tch::Kind::Float, self.device.unwrap())));
            let output1 = self.model.as_ref().unwrap().forward_is(&[inputs_ival, length_ival, h0]).unwrap();
            let (_, hidden) = match output1 {
                tch::IValue::Tuple(ivalues) => match &ivalues[..] {
                    [tch::IValue::Tensor(t1), tch::IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
                    _ => panic!("unexpected output {:?}", ivalues),
                },
                _ => panic!("unexpected output {:?}", output1),
            };
            // println!("hidden : {:?}", hidden);
            // last word
            let last_word = Tensor::of_slice(&[inputs.last().copied().unwrap() as i64]).to_device(self.device.unwrap());
            let last_word = tch::IValue::Tensor(Tensor::stack(&[last_word], 0).to_device(self.device.unwrap()));
            let length = tch::IValue::Tensor(Tensor::of_slice(&[1]).to_device(self.device.unwrap()));
            let hidden = tch::IValue::Tensor(hidden);
            let output2 = self.model.as_ref().unwrap().forward_is(&[last_word, length, hidden]).unwrap();
            let (output, _) = match output2 {
                tch::IValue::Tuple(ivalues) => match &ivalues[..] {
                    [tch::IValue::Tensor(t1), tch::IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
                    _ => panic!("unexpected output {:?}", ivalues),
                },
                _ => panic!("unexpected output {:?}", output2),
            };
            Some(output)
        }
    }

    pub fn exists(&self) -> bool {
        self.model.is_some()
    }
}

pub fn generate(model: &Model, calls: &[u32], rng: &mut RngType) -> i32 {
    let topk = 10;
    let mut prev_calls: Vec<SyscallId> = calls.iter().map(|c| *c as usize+3).collect();
    // "2" refer to Start-Of-Sentence(SOS)
    prev_calls.insert(0, 2);
    let prev_pred = model.eval(&prev_calls).unwrap();
    let candidates = top_k(&prev_pred, topk);
    let candidates: Vec<(SyscallId, f64)> = candidates.into_iter().collect::<Vec<(SyscallId, f64)>>();
    if let Some(sid) = candidates
        .choose_weighted(rng, |candidate| candidate.1)
        .ok()
        .map(|candidate| candidate.0) {
        if sid >= 3 {
            return sid as i32-3;
        }
        else {
            return -1;
        }
    }
    -1
}

pub fn mutate(model: &Model, calls: &[u32], rng: &mut RngType, idx: usize) -> i32 {
    // first, consider calls that can be influenced by calls before `idx`.
    // as in NLP, syscall are labeled(word_id) from "3", 0->3, 1->4,
    // so it require to cast between sid and word_id
    let topk = 10;
    let mut prev_calls: Vec<SyscallId> = calls[..idx].iter().map(|c| *c as usize+3).collect();
    // "1" refer to Start-Of-Sentence(SOS)
    prev_calls.insert(0, 1);
    let prev_pred = model.eval(&prev_calls).unwrap();
    let mut candidates = top_k(&prev_pred, topk);

    // then, consider calls that can be influence calls after `idx`.
    // cast between sid and word_id
    if idx != calls.len() {
        let mut back_calls: Vec<SyscallId> = calls[idx..].iter().rev().map(|c| *c as usize+3).collect();
        // "2" refer to End-of-Sentence(EOS)
        back_calls.insert(0, 2);
        let back_pred = model.eval(&back_calls).unwrap();
        candidates.extend(top_k(&back_pred, topk));
    }

    let candidates: Vec<(SyscallId, f64)> = candidates.into_iter().collect::<Vec<(SyscallId, f64)>>();
    if let Ok(candidate) = candidates.choose_weighted(rng, |candidate| candidate.1) {
        if candidate.0 >= 3 {
            return candidate.0 as i32 -3;
        }
        else {
            // failed to select with relation, use normal strategy.
            return -1;
        }
    }
    -1
}

// generate `topk` candidates for given distribution
pub fn top_k(pred: &Tensor, topk: i64) -> HashMap<usize, f64>{
    let (pred_val, pred_indexes) = pred.topk(topk, 1, true, true);
    let pred_val: Vec<f64> = Vec::from(pred_val);
    let pred_indexes: Vec<i16> = Vec::from(pred_indexes);
    let candidates: HashMap<SyscallId, f64> = pred_indexes.into_iter()
            .map(|i| i as SyscallId)
            .zip(pred_val.into_iter())
            .collect();
    candidates
}

#[no_mangle]
pub extern "C" fn model_default() -> *mut Model {
    Box::into_raw(Box::new(Model::default()))
}

#[no_mangle]
pub extern "C" fn model_new(model_path: *const c_char, device_id: u32) -> *mut Model {
    let path_c_str = unsafe {
        assert!(!model_path.is_null());
        CStr::from_ptr(model_path)
    };

    let path_r_str = path_c_str.to_str().unwrap();
    Box::into_raw(Box::new(Model::new(path_r_str, device_id as usize)))
}

#[no_mangle]
pub extern "C" fn model_load(ptr: *mut Model, model_path: *const c_char, device_id: u32) {
    let model = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    
    let path_c_str = unsafe {
        assert!(!model_path.is_null());
        CStr::from_ptr(model_path)
    };

    let path_r_str = path_c_str.to_str().unwrap();
    model.load(path_r_str, device_id as usize)
}

#[no_mangle]
pub extern "C" fn model_mutate(ptr: *mut Model, calls_raw: *const u32, len: size_t, idx: u32) -> i32  {
    let model = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };

    let calls = unsafe {
        assert!(!calls_raw.is_null());
        slice::from_raw_parts(calls_raw, len as usize)
    };

    let mut rng = SmallRng::from_entropy();
    mutate(model, calls, &mut rng, idx as usize)
}

#[no_mangle]
pub extern "C" fn model_gen(ptr: *mut Model, calls_raw: *const u32, len: size_t) -> i32  {
    let model = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };

    let calls = unsafe {
        assert!(!calls_raw.is_null());
        slice::from_raw_parts(calls_raw, len as usize)
    };

    let mut rng = SmallRng::from_entropy();
    generate(model, calls, &mut rng)
}

#[no_mangle]
pub extern "C" fn model_exists(ptr: *mut Model) -> u32 {
    let model = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    return model.exists() as u32
}

#[cfg(test)]
mod tests {
    use crate::Model;
    use tch::Device;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_cuda() {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(0);
        }
        println!("{:?}", device);
        assert!(device.is_cuda());
    }

    #[test]
    fn test_load_model() {
        let mut model = Model::default();
        assert!(!model.exists());
        let model_file = "/home/workdir/syscall_model_jit_best.pt";
        model.load(model_file, 0);
        assert!(model.exists());
    }

    #[test]
    fn test_eval_model() {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(1);
        }
        println!("{:?}", device);
        assert!(device.is_cuda());
        let model = Model::new("/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt", 1);
        let out = model.eval(&[0, 3920, 3276]).unwrap().to(device);
        println!("{:?}", out);
        assert_eq!(out.size(), vec![1,4284])
    }
}
