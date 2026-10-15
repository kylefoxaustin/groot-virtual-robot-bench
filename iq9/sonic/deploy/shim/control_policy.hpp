// control_policy.hpp — QNN replacement of upstream PolicyEngine (the decoder; same class name and public API;
// groot-virtual-robot-bench, not upstream). SONIC_CPU_DEC_ONNX=<model_decoder.onnx> runs it on the CPU instead (control arm).
#ifndef CONTROL_POLICY_HPP
#define CONTROL_POLICY_HPP
#include <string>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <TRTInference/InferenceEngine.h>
#include "qnn_shim_rt.hpp"
#include <cstdlib>
class PolicyEngine {
 public:
  bool Initialize(const std::string& model_path, bool use_fp16 = false) {
    if (const char* cpu = std::getenv("SONIC_CPU_DEC_ONNX")) {  // control arm: upstream decoder ONNX on the A78, ORT fp32
      Ort::SessionOptions so; so.SetIntraOpNumThreads(qshim::envi("SONIC_CPU_THREADS", 4)); so.SetInterOpNumThreads(1);
      cpu_.reset(new Ort::Session(qshim::Runtime::Get().OrtEnv(), cpu, so));
      input_dimension_ = 994; action_dimension_ = 29; policy_input_buffer_.assign(994, 0.f); action_buffer_.assign(29, 0.f);
      initialized_ = true; std::cout << "✓ Policy engine initialized (CPU ORT fp32: " << cpu << ")" << std::endl; return true;
    }
    auto& R = qshim::Runtime::Get(); if (!R.Init()) return false;
    std::string bin = qshim::envs("SONIC_QNN_DEC"); int dev = qshim::envi("SONIC_QNN_DEC_DEV", 0);
    if (!R.Load(bin, dev, G_)) { std::cerr << "✗ [qnn-shim] decoder load failed: " << bin << std::endl; return false; }
    if (G_.nIn() != 1 || G_.nOut() != 1) return false;
    input_dimension_ = qshim::telems(G_.inputs[0]); action_dimension_ = qshim::telems(G_.outputs[0]);
    policy_input_buffer_.assign(input_dimension_, 0.f); action_buffer_.assign(action_dimension_, 0.f);
    initialized_ = true;
    std::cout << "✓ Policy engine initialized (QNN, onnx " << model_path << " -> " << bin << ", device " << dev << ", in " << input_dimension_ << ", act " << action_dimension_ << ")" << std::endl;
    return true;
  }
  bool CaptureGraph() { return initialized_; }
  bool Infer(cudaStream_t = nullptr) {
    if (!initialized_) return false;
    if (cpu_) {
      int64_t ish[2] = {1, (int64_t)input_dimension_}; auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value in = Ort::Value::CreateTensor<float>(mi, policy_input_buffer_.data(), input_dimension_, ish, 2);
      const char* inn[] = {"obs_dict"}; const char* on[] = {"action"};
      auto o = cpu_->Run(Ort::RunOptions{nullptr}, inn, &in, 1, on, 1);
      memcpy(action_buffer_.data(), o[0].GetTensorData<float>(), action_dimension_ * 4); return true;
    }
    memcpy(qshim::fbuf(G_.inputs[0]), policy_input_buffer_.data(), input_dimension_ * 4);
    if (!qshim::Runtime::Get().Exec(G_)) return false;
    memcpy(action_buffer_.data(), qshim::fbuf(G_.outputs[0]), action_dimension_ * 4);
    return true;
  }
  size_t GetInputDimension() const { return input_dimension_; }
  size_t GetActionDimension() const { return action_dimension_; }
  bool IsInitialized() const { return initialized_; }
  TPinnedVector<float>& GetInputBuffer() { return policy_input_buffer_; }
  TPinnedVector<float>& GetActionBuffer() { return action_buffer_; }
 private:
  qshim::Graph G_; size_t input_dimension_ = 0, action_dimension_ = 0; bool initialized_ = false; std::unique_ptr<Ort::Session> cpu_;
  TPinnedVector<float> policy_input_buffer_, action_buffer_;
};
#endif
