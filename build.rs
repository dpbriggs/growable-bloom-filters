fn main() {
    {
        prost_build::compile_protos(&["proto/growable_bloom.proto"], &["proto"]).unwrap();
    }
}