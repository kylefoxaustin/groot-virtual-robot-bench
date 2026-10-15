// encoder.hpp — QNN replacement of upstream EncoderEngine (same class name and public API; groot-virtual-robot-bench, not upstream).
// The NPU (SONIC_QNN_ENC_DEV) runs the 'sonic_enc_core' graph (all 3 sub-encoders up to the pre-Round value [3,64],
// iq9/sonic/encdec/make_graphs.py); the CPU does the FSQ round()/16 and the encoder-mode select (mode = obs_dict[0]).
// SONIC_CPU_ENC_ONNX=<model_encoder.onnx> instead runs the upstream encoder on the CPU (onnxruntime fp32): the CPU control arm.
#ifndef ENCODER_HPP
#define ENCODER_HPP
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <TRTInference/InferenceEngine.h>
#include "qnn_shim_rt.hpp"
#include <cstdlib>
class EncoderEngine {
 public:
  bool Initialize(const std::string& model_path, bool use_fp16 = false) {
    if (const char* cpu = std::getenv("SONIC_CPU_ENC_ONNX")) {  // control arm: full upstream ONNX on the A78, ORT fp32
      Ort::SessionOptions so; so.SetIntraOpNumThreads(qshim::envi("SONIC_CPU_THREADS", 4)); so.SetInterOpNumThreads(1);
      cpu_.reset(new Ort::Session(qshim::Runtime::Get().OrtEnv(), cpu, so));
      input_dimension_ = 1762; encoder_input_buffer_.assign(input_dimension_, 0.f); token_buffer_.assign(64, 0.f);
      initialized_ = true; std::cout << "✓ Encoder initialized (CPU ORT fp32: " << cpu << ")" << std::endl; return true;
    }
    auto& R = qshim::Runtime::Get(); if (!R.Init()) return false;
    std::string bin = qshim::envs("SONIC_QNN_ENC"); int dev = qshim::envi("SONIC_QNN_ENC_DEV", 0);
    if (!R.Load(bin, dev, G_)) { std::cerr << "✗ [qnn-shim] encoder load failed: " << bin << std::endl; return false; }
    input_dimension_ = qshim::telems(G_.inputs[0]); size_t o = qshim::telems(G_.outputs[0]);
    if (G_.nIn() != 1 || G_.nOut() != 1 || o != 192) { std::cerr << "✗ [qnn-shim] encoder core must be 1 in / 192 out" << std::endl; return false; }
    encoder_input_buffer_.assign(input_dimension_, 0.f); token_buffer_.assign(64, 0.f);
    initialized_ = true;
    std::cout << "✓ Encoder initialized (QNN, onnx " << model_path << " -> " << bin << ", device " << dev << ", in " << input_dimension_ << ", tokens 64)" << std::endl;
    return true;
  }
  bool CaptureGraph() { return initialized_; }
  bool Encode(cudaStream_t = nullptr) {
    if (!initialized_) return false;
    if (cpu_) {
      int64_t ish[2] = {1, (int64_t)input_dimension_}; auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value in = Ort::Value::CreateTensor<float>(mi, encoder_input_buffer_.data(), input_dimension_, ish, 2);
      const char* inn[] = {"obs_dict"}; const char* on[] = {"encoded_tokens"};
      auto o = cpu_->Run(Ort::RunOptions{nullptr}, inn, &in, 1, on, 1);
      memcpy(token_buffer_.data(), o[0].GetTensorData<float>(), 64 * 4); return true;
    }
    memcpy(qshim::fbuf(G_.inputs[0]), encoder_input_buffer_.data(), input_dimension_ * 4);
    if (!qshim::Runtime::Get().Exec(G_)) return false;
    const float* eo = qshim::fbuf(G_.outputs[0]);
    int k = (int)encoder_input_buffer_[0]; if (k < 0 || k > 2) k = 0;
    for (int j = 0; j < 64; ++j) token_buffer_[j] = std::nearbyint(eo[k * 64 + j]) / 16.0f;  // FSQ round-half-even /16
    return true;
  }
  size_t GetInputDimension() const { return input_dimension_; }
  size_t GetTokenDimension() const { return 64; }
  bool IsInitialized() const { return initialized_; }
  TPinnedVector<float>& GetInputBuffer() { return encoder_input_buffer_; }
  TPinnedVector<float>& GetTokenBuffer() { return token_buffer_; }
 private:
  qshim::Graph G_; size_t input_dimension_ = 0; bool initialized_ = false; std::unique_ptr<Ort::Session> cpu_;
  TPinnedVector<float> encoder_input_buffer_, token_buffer_;
};
#endif
