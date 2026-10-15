// localmotion_kplanner_tensorrt.hpp — QNN split-planner replacement of upstream LocalMotionPlannerTensorRT
// (same class name + constructor so the deploy main is unchanged; groot-virtual-robot-bench, not upstream).
// Transformer/conv blocks S,R,P,D: QNN fp16 on SONIC_QNN_PLANNER_DEV; ArgMax/TopK/OneHot/CumSum/ScatterND/seed hash: CPU ORT fp32.
#ifndef LOCALMOTION_KPLANNER_TENSORRT_HPP
#define LOCALMOTION_KPLANNER_TENSORRT_HPP
#include <TRTInference/InferenceEngine.h>
#include <iostream>
#include "localmotion_kplanner.hpp"
#include <cuda_runtime.h>
#include "qnn_shim_rt.hpp"
class LocalMotionPlannerTensorRT : public LocalMotionPlannerBase {
 public:
  LocalMotionPlannerTensorRT(bool use_fp16 = false, int device_id = 0, const PlannerConfig& config = PlannerConfig())
      : LocalMotionPlannerBase(config) {
    context_.assign(4 * 36, 0.f); qpos_.assign(64 * 36, 0.f);
    std::string dir = qshim::envs("SONIC_QNN_PLANNER_DIR"); int dev = qshim::envi("SONIC_QNN_PLANNER_DEV", 1);
    if (!sp_.Load(dir, dev, qshim::envi("SONIC_QNN_PLANNER_THREADS", 4))) throw std::runtime_error("QNN split planner load failed");
    std::cout << "✓ [qnn-shim] split planner loaded from " << dir << " on HTP deviceId " << dev << " (onnx " << config.model_path << ")" << std::endl;
  }
  bool InitializeSpecific() override { return sp_.loaded; }
  void RunInference() override {
    if (!sp_.Run(context_.data(), mode_, target_vel_, mv_, face_, seed_, height_, allowed_, qpos_.data(), npf_))
      throw std::runtime_error("QNN split planner inference failed");
  }
  void UpdateInputTensors(int mode_value, float target_vel, float target_height, const std::array<float, 3>& movement_direction,
                          const std::array<float, 3>& facing_direction, int random_seed) override {
    if (mode_value < 0 || mode_value >= GetValidModeValueRange()) std::cout << "✗ Invalid mode value: " << mode_value << std::endl;
    mode_ = mode_value < GetValidModeValueRange() ? mode_value : 0;
    target_vel_ = target_vel;
    if (config_.version == 1 || config_.version == 2) height_ = target_height;
    for (int i = 0; i < 3; ++i) { mv_[i] = movement_direction[i]; face_[i] = facing_direction[i]; }
    if (random_seed != -1) { current_random_seed_ = random_seed; seed_ = random_seed; }
  }
  float* GetContextBuffer() override { return context_.data(); }
  int32_t GetNumPredFrames() override { return npf_; }
  const float* GetMujocoQposBuffer() override { return qpos_.data(); }
  float* GetMovementDirectionValues() override { return mv_; }
  float* GetFacingDirectionValues() override { return face_; }
  float GetTargetVelValue() override { return target_vel_; }
  float GetHeightValue() override { return height_; }
  int32_t GetRandomSeedValue() override { return (int32_t)seed_; }
  int32_t GetModeValue() override { return (int32_t)mode_; }
 private:
  qshim::SplitPlanner sp_;
  std::vector<float> context_, qpos_;
  int64_t mode_ = 0, seed_ = 0; float target_vel_ = -1.f, height_ = -1.f; float mv_[3] = {0, 0, 0}, face_[3] = {0, 0, 0};
  int32_t npf_ = 0;
  int64_t allowed_[11] = {0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0};  // upstream TRT planner default mask
};
#endif
