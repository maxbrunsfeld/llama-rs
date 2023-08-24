use llama_rs::{Context, Model, Params, Token};
use std::{fs, time::Instant};

fn main() {
    let params = Params::new()
        .embedding_only()
        .gpu_layers(16)
        .batch_size(1024)
        .context_size(2048);

    let model = Model::new("../llama.cpp/models/llama-2-7b.gguf".as_ref(), params).unwrap();
    let mut llama = Context::new(&model, params).unwrap();
    let mut tokens = vec![0; 2048];

    let filenames = std::env::args().skip(1);
    for filename in filenames {
        let text = fs::read_to_string(&filename).expect("failed to read file");

        for _ in 0..10 {
            let t0 = Instant::now();
            let embeddings = &get_embeddings(&mut llama, &mut tokens, &text)[0..5];
            println!("{filename:?}: time: {:?}, {embeddings:?}", t0.elapsed());
        }
    }
}

fn get_embeddings<'a>(
    llama: &'a mut Context,
    token_buffer: &mut Vec<Token>,
    text: &str,
) -> &'a [f32] {
    let len = llama.tokenize(text, token_buffer, true).unwrap();
    token_buffer.truncate(len);
    llama.eval(&token_buffer, 0, 1).unwrap();
    llama.embeddings()
}
