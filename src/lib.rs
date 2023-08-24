use anyhow::{anyhow, Result};
use std::{
    ffi::{c_char, CString},
    path::Path,
};

mod bindings;
use bindings as c;

static INIT: std::sync::Once = std::sync::Once::new();

pub struct Model(*mut c::llama_model);
pub struct Context(*mut c::llama_context);
pub type Token = c::llama_token;

#[derive(Copy, Clone)]
pub struct Params(c::llama_context_params);

impl Params {
    pub fn new() -> Self {
        Self(unsafe { c::llama_context_default_params() })
    }

    pub fn gpu_layers(mut self, count: usize) -> Self {
        self.0.n_gpu_layers = count as i32;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.0.n_batch = size as i32;
        self
    }

    pub fn embedding_only(mut self) -> Self {
        self.0.embedding = true;
        self
    }

    pub fn context_size(mut self, size: usize) -> Self {
        self.0.n_ctx = size as i32;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new(path: &Path, params: Params) -> Result<Self> {
        unsafe {
            INIT.call_once(|| c::llama_backend_init(true));

            let path = CString::new(path.to_str().ok_or_else(|| anyhow!("invalid path"))?)?;
            let ptr = c::llama_load_model_from_file(path.as_ptr(), params.0);
            if ptr.is_null() {
                Err(anyhow::anyhow!("Failed to load model"))
            } else {
                Ok(Self(ptr))
            }
        }
    }

    pub fn model_type(&self) -> String {
        unsafe {
            let mut result = vec![0u8; 256];
            let len = c::llama_model_type(self.0, result.as_ptr() as *mut c_char, result.len());
            result.truncate(len.max(0) as usize);
            String::from_utf8(result).unwrap_or(String::new())
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            c::llama_free_model(self.0);
        }
    }
}

impl Context {
    pub fn new(model: &Model, params: Params) -> Result<Self> {
        unsafe {
            let ptr = c::llama_new_context_with_model(model.0, params.0);
            if ptr.is_null() {
                Err(anyhow!("failed to create context"))
            } else {
                Ok(Self(ptr))
            }
        }
    }

    pub fn tokenize(&self, text: &str, output: &mut [Token], is_start: bool) -> Result<usize> {
        let text = CString::new(text)?;
        unsafe {
            let len = c::llama_tokenize(
                self.0,
                text.as_ptr(),
                output.as_mut_ptr(),
                output.len() as i32,
                is_start,
            );
            if len > 0 {
                Ok(len as usize)
            } else {
                Err(anyhow!("failed to tokenize"))
            }
        }
    }

    pub fn eval(&mut self, tokens: &[Token], n_past: usize, n_threads: usize) -> Result<()> {
        unsafe {
            let code = c::llama_eval(
                self.0,
                tokens.as_ptr(),
                tokens.len() as i32,
                n_past as i32,
                n_threads as i32,
            );
            if code == 0 {
                Ok(())
            } else {
                Err(anyhow!("failed to eval"))
            }
        }
    }

    pub fn embeddings(&self) -> &[f32] {
        unsafe {
            let len = c::llama_n_embd(self.0);
            let ptr = c::llama_get_embeddings(self.0);
            std::slice::from_raw_parts(ptr, len as usize)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            c::llama_free(self.0);
        }
    }
}
