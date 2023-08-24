use bindgen::Builder;
use std::{env, fs, path::PathBuf};

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    Builder::default()
        .clang_args(["-D", "LLAMA_METAL=1", "-xc++"])
        .header("./vendor/llama.cpp/llama.h")
        .layout_tests(false)
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        // .write_to_file("src/bindings.rs")
        .expect("failed to write bindings");

    fs::create_dir_all("target/debug").unwrap();
    fs::copy(
        "./vendor/llama.cpp/ggml-metal.metal",
        "./target/debug/ggml-metal.metal",
    )
    .unwrap();

    println!("cargo:rustc-link-lib=framework=System");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    cc::Build::new()
        .cpp(true)
        .object("./vendor/llama.cpp/common.o")
        .object("./vendor/llama.cpp/ggml-alloc.o")
        .object("./vendor/llama.cpp/ggml-metal.o")
        .object("./vendor/llama.cpp/ggml.o")
        .object("./vendor/llama.cpp/k_quants.o")
        .object("./vendor/llama.cpp/llama.o")
        .compile("binding");
}
