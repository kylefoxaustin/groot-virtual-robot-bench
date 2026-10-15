// qnn_shim_rt.hpp — QNN HTP runtime + split-planner chain for the IQ-9075 deploy shim (groot-virtual-robot-bench; not upstream).
// Replaces TensorRT in gear_sonic_deploy's g1_deploy_onnx_ref for the IQ-9075: encoder/decoder/planner run as QNN fp16 context binaries
// on a Hexagon NSP chosen per network (strict deviceId pinning), planner data-dependent ops on CPU (onnxruntime).
// Config via environment (hard error if missing):
//   SONIC_QNN_BACKEND, SONIC_QNN_SYSLIB      libQnnHtp.so / libQnnSystem.so
//   SONIC_QNN_ENC, SONIC_QNN_DEC             encoder-core / decoder context bins (iq9/sonic/encdec)
//   SONIC_QNN_PLANNER_DIR                    host_H0..4.onnx, manifest.txt, planner_{S,R,P,D}_aaf0_fp16.bin
//   SONIC_QNN_ENC_DEV, SONIC_QNN_DEC_DEV, SONIC_QNN_PLANNER_DEV   (default 0,0,1)
//   SONIC_QNN_PERF (burst|none, default burst), SONIC_QNN_PLANNER_THREADS (ORT intra-op threads of the planner host stages,
//   default 4; the IQ-9075 late-start settings use 1)
// The Qualcomm headers below come from your QAIRT SDK (include/QNN and the SampleApp sources); none are in this repo.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "DynamicLoadUtil.hpp"
#include "IOTensor.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleAppUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnWrapperUtils.hpp"
#include "SampleApp.hpp"
#include "DataUtil.hpp"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnDevice.h"
#include <onnxruntime_cxx_api.h>

namespace qshim {

inline std::string envs(const char* k, const char* def = nullptr) {
  const char* v = std::getenv(k);
  if (v && *v) return v;
  if (def) return def;
  std::cerr << "✗ [qnn-shim] missing env " << k << std::endl;
  throw std::runtime_error(std::string("missing env ") + k);
}
inline int envi(const char* k, int def) { const char* v = std::getenv(k); return (v && *v) ? std::atoi(v) : def; }

struct Graph {
  std::string name; int device = -1;
  Qnn_ContextHandle_t context = nullptr;
  qnn_wrapper_api::GraphInfo_t** graphsInfo = nullptr; uint32_t graphsCount = 0;
  Qnn_Tensor_t *inputs = nullptr, *outputs = nullptr;
  uint32_t nIn() const { return (*graphsInfo)[0].numInputTensors; }
  uint32_t nOut() const { return (*graphsInfo)[0].numOutputTensors; }
};
inline float* fbuf(Qnn_Tensor_t& t) { return static_cast<float*>(QNN_TENSOR_GET_CLIENT_BUF(t).data); }
inline size_t telems(Qnn_Tensor_t& t) {
  uint32_t r = QNN_TENSOR_GET_RANK(t); uint32_t* d = QNN_TENSOR_GET_DIMENSIONS(t);
  size_t n = 1; for (uint32_t i = 0; i < r; ++i) n *= d[i]; return n;
}
inline std::string tname(Qnn_Tensor_t& t) { return QNN_TENSOR_GET_NAME(t); }

class Runtime {
 public:
  static Runtime& Get() { static Runtime r; return r; }
  bool Init() {
    std::lock_guard<std::mutex> l(m_);
    if (ok_) return true;
    qnn::log::initializeLogging(); qnn::log::setLogLevel(QNN_LOG_LEVEL_ERROR);
    std::string be = envs("SONIC_QNN_BACKEND"), sl = envs("SONIC_QNN_SYSLIB");
    if (qnn::tools::dynamicloadutil::getQnnFunctionPointers(be, "", &fp_, &beh_, false, &mh_) != qnn::tools::dynamicloadutil::StatusCode::SUCCESS) return false;
    if (qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(sl, &fp_) != qnn::tools::dynamicloadutil::StatusCode::SUCCESS) return false;
    if (fp_.qnnInterface.logCreate) fp_.qnnInterface.logCreate(qnn::log::getLogCallback(), qnn::log::getLogLevel(), &log_);
    if (fp_.qnnInterface.backendCreate(log_, nullptr, &backend_) != QNN_BACKEND_NO_ERROR) return false;
    ok_ = true; std::cout << "✓ [qnn-shim] QNN HTP backend up" << std::endl;
    return true;
  }
  bool Device(int id) {
    if (dev_.count(id)) return true;
    const QnnDevice_PlatformInfo_t* pi = nullptr;
    if (fp_.qnnInterface.deviceGetPlatformInfo(log_, &pi) != QNN_SUCCESS || !pi) return false;
    const QnnDevice_HardwareDeviceInfo_t* hit = nullptr;
    for (uint32_t i = 0; i < pi->v1.numHwDevices; ++i) if ((int)pi->v1.hwDevices[i].v1.deviceId == id) hit = &pi->v1.hwDevices[i];
    if (!hit) { std::cerr << "✗ [qnn-shim] no HTP deviceId " << id << " (strict, no fallback)" << std::endl; return false; }
    auto* s = new Sel(); s->hw = *hit;
    s->pi.version = QNN_DEVICE_PLATFORM_INFO_VERSION_1; s->pi.v1.numHwDevices = 1; s->pi.v1.hwDevices = &s->hw;
    s->cfg.option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO; s->cfg.hardwareInfo = &s->pi; s->cfgp[0] = &s->cfg; s->cfgp[1] = nullptr;
    Qnn_DeviceHandle_t h = nullptr;
    if (fp_.qnnInterface.deviceCreate(log_, s->cfgp, &h) != QNN_SUCCESS) return false;
    dev_[id] = h; sels_.emplace_back(s);
    std::cout << "✓ [qnn-shim] HTP deviceId " << id << " created" << std::endl;
    SetPerf(id, envs("SONIC_QNN_PERF", "burst"));
    return true;
  }
  bool Load(const std::string& path, int dev, Graph& G) {
    std::lock_guard<std::mutex> l(m_);
    if (!Device(dev)) return false;
    qnn::tools::datautil::StatusCode ds; uint64_t sz = 0;
    std::tie(ds, sz) = qnn::tools::datautil::getFileSize(path);
    if (sz == 0) { std::cerr << "✗ [qnn-shim] empty/missing " << path << std::endl; return false; }
    std::vector<uint8_t> buf(sz);
    if (qnn::tools::datautil::readBinaryFromFile(path, buf.data(), sz) != qnn::tools::datautil::StatusCode::SUCCESS) return false;
    QnnSystemContext_Handle_t sys = nullptr;
    if (fp_.qnnSystemInterface.systemContextCreate(&sys) != QNN_SUCCESS) return false;
    const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
    if (fp_.qnnSystemInterface.systemContextGetBinaryInfo(sys, buf.data(), sz, &bi, &bis) != QNN_SUCCESS) return false;
    if (!qnn::tools::sample_app::copyMetadataToGraphsInfo(bi, G.graphsInfo, G.graphsCount)) return false;
    fp_.qnnSystemInterface.systemContextFree(sys);
    if (fp_.qnnInterface.contextCreateFromBinary(backend_, dev_[dev], nullptr, buf.data(), sz, &G.context, nullptr) != QNN_SUCCESS) return false;
    if (fp_.qnnInterface.graphRetrieve(G.context, (*G.graphsInfo)[0].graphName, &((*G.graphsInfo)[0].graph)) != QNN_SUCCESS) return false;
    if (io_.setupInputAndOutputTensors(&G.inputs, &G.outputs, (*G.graphsInfo)[0]) != qnn::tools::iotensor::StatusCode::SUCCESS) return false;
    G.name = (*G.graphsInfo)[0].graphName; G.device = dev;
    for (uint32_t i = 0; i < G.nIn(); ++i) if (QNN_TENSOR_GET_DATA_TYPE(G.inputs[i]) != QNN_DATATYPE_FLOAT_32) return false;
    for (uint32_t i = 0; i < G.nOut(); ++i) if (QNN_TENSOR_GET_DATA_TYPE(G.outputs[i]) != QNN_DATATYPE_FLOAT_32) return false;
    std::cout << "✓ [qnn-shim] loaded " << G.name << " on HTP deviceId " << dev << " (" << sz / 1048576 << " MB)" << std::endl;
    return true;
  }
  bool Exec(Graph& G) {
    return fp_.qnnInterface.graphExecute((*G.graphsInfo)[0].graph, G.inputs, G.nIn(), G.outputs, G.nOut(), nullptr, nullptr) == QNN_GRAPH_NO_ERROR;
  }
  Ort::Env& OrtEnv() { static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "sonic_qnn_shim"); return e; }

 private:
  struct Sel { QnnDevice_HardwareDeviceInfo_t hw; QnnDevice_PlatformInfo_t pi; QnnDevice_Config_t cfg; const QnnDevice_Config_t* cfgp[2]; };
  void SetPerf(int dev, const std::string& mode) {
    if (mode == "none") { std::cout << "[qnn-shim] perf vote none on device " << dev << std::endl; return; }
    QnnDevice_Infrastructure_t inf = nullptr;
    if (fp_.qnnInterface.deviceGetInfrastructure(&inf) != QNN_SUCCESS) return;
    auto& pi = static_cast<QnnHtpDevice_Infrastructure_t*>(inf)->perfInfra;
    uint32_t id = 0;
    if (pi.createPowerConfigId((uint32_t)dev, 0, &id) != QNN_SUCCESS) { std::cerr << "⚠ [qnn-shim] createPowerConfigId failed dev " << dev << std::endl; return; }
    QnnHtpPerfInfrastructure_PowerConfig_t dc; memset(&dc, 0, sizeof(dc));
    dc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& v = dc.dcvsV3Config;
    v.contextId = id; v.setDcvsEnable = 1; v.dcvsEnable = 0;
    v.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    v.setSleepLatency = 1; v.sleepLatency = 40; v.setSleepDisable = 1; v.sleepDisable = 1;
    v.setBusParams = 1; v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v.setCoreParams = 1; v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dc, nullptr};
    Qnn_ErrorHandle_t e = pi.setPowerConfig(id, cfgs);
    std::cout << "[qnn-shim] perf vote " << mode << " on device " << dev << " rc=" << e << std::endl;
  }
  std::mutex m_; bool ok_ = false;
  qnn::tools::sample_app::QnnFunctionPointers fp_; void *beh_ = nullptr, *mh_ = nullptr;
  Qnn_LogHandle_t log_ = nullptr; Qnn_BackendHandle_t backend_ = nullptr;
  std::map<int, Qnn_DeviceHandle_t> dev_; std::vector<std::unique_ptr<Sel>> sels_;
  qnn::tools::iotensor::IOTensor io_;
};

inline std::string sanitize(const std::string& s) { std::string o = s; for (auto& c : o) if (!isalnum((unsigned char)c) && c != '_') c = '_'; return o; }

struct T { ONNXTensorElementDataType dt; std::vector<int64_t> shape; std::vector<uint8_t> data; };
inline size_t esize(ONNXTensorElementDataType dt) {
  switch (dt) { case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
    default: return 0; }
}
inline float asFloat(const T& t, size_t i) {
  switch (t.dt) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return ((const float*)t.data.data())[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return (float)((const int64_t*)t.data.data())[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return (float)((const int32_t*)t.data.data())[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return ((const uint8_t*)t.data.data())[i] ? 1.f : 0.f;
    default: return NAN; }
}

// Split planner: H0 S H1 R H2 P H3 D H4 (host ORT fp32 stages, QNN fp16 blocks)
struct SplitPlanner {
  std::vector<std::unique_ptr<Ort::Session>> H; std::vector<std::vector<std::string>> Hin, Hout;
  Graph G[4]; std::vector<std::string> Mins[4], Mouts[4]; std::vector<int> inMap[4], outMap[4];
  std::map<std::string, T> envt; Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  bool loaded = false;
  bool Load(const std::string& dir, int dev, int threads) {
    auto& R = Runtime::Get(); if (!R.Init()) return false;
    Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetInterOpNumThreads(1);
    std::cerr << "[qnn-shim] planner ORT intra_op_threads=" << threads << std::endl;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    for (int k = 0; k < 5; ++k) {
      std::string p = dir + "/host_H" + std::to_string(k) + ".onnx";
      H.emplace_back(new Ort::Session(R.OrtEnv(), p.c_str(), so));
      Ort::AllocatorWithDefaultOptions al; std::vector<std::string> in, out;
      for (size_t i = 0; i < H[k]->GetInputCount(); ++i) in.push_back(H[k]->GetInputNameAllocated(i, al).get());
      for (size_t i = 0; i < H[k]->GetOutputCount(); ++i) out.push_back(H[k]->GetOutputNameAllocated(i, al).get());
      Hin.push_back(in); Hout.push_back(out);
    }
    std::ifstream mf(dir + "/manifest.txt"); std::string a, b; int cur = -1;
    while (mf >> a >> b) { if (a == "block") cur++; else if (a == "in") Mins[cur].push_back(b); else if (a == "out") Mouts[cur].push_back(b); }
    if (cur != 3) return false;
    const char* bn[4] = {"S", "R", "P", "D"};
    for (int k = 0; k < 4; ++k) {
      if (!R.Load(dir + "/planner_" + bn[k] + "_aaf0_fp16.bin", dev, G[k])) return false;
      for (uint32_t i = 0; i < G[k].nIn(); ++i) {
        std::string q = tname(G[k].inputs[i]); int hit = -1;
        for (size_t j = 0; j < Mins[k].size(); ++j) if (sanitize(Mins[k][j]) == q || sanitize(Mins[k][j] + "__f32") == q) hit = (int)j;
        if (hit < 0) return false; inMap[k].push_back(hit);
      }
      for (uint32_t i = 0; i < G[k].nOut(); ++i) {
        std::string q = tname(G[k].outputs[i]); int hit = -1;
        for (size_t j = 0; j < Mouts[k].size(); ++j) if (sanitize(Mouts[k][j]) == q) hit = (int)j;
        if (hit < 0) return false; outMap[k].push_back(hit);
      }
    }
    loaded = true; return true;
  }
  void setF(const std::string& n, std::vector<int64_t> sh, const float* v) {
    T t; t.dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; t.shape = sh; size_t ne = 1; for (auto s : sh) ne *= s;
    t.data.resize(ne * 4); memcpy(t.data.data(), v, ne * 4); envt[n] = std::move(t);
  }
  void setI64(const std::string& n, std::vector<int64_t> sh, std::vector<int64_t> v) {
    T t; t.dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; t.shape = sh; t.data.resize(v.size() * 8); memcpy(t.data.data(), v.data(), v.size() * 8); envt[n] = std::move(t);
  }
  bool Run(const float* ctx, int64_t mode, float tv, const float* mv, const float* face, int64_t seed, float height,
           const int64_t* allowed11, float* qpos_out, int32_t& npf_out) {
    envt.clear();
    setF("context_mujoco_qpos", {1, 4, 36}, ctx); setI64("mode", {1}, {mode}); setF("target_vel", {1}, &tv);
    setF("movement_direction", {1, 3}, mv); setF("facing_direction", {1, 3}, face); setI64("random_seed", {1}, {seed});
    setF("height", {1}, &height); setI64("has_specific_target", {1, 1}, {0});
    float z[12] = {0}; setF("specific_target_positions", {1, 4, 3}, z); setF("specific_target_headings", {1, 4}, z);
    setI64("allowed_pred_num_tokens", {1, 11}, std::vector<int64_t>(allowed11, allowed11 + 11));
    for (int k = 0; k < 5; ++k) {
      if (!RunHost(k)) return false;
      if (k < 4 && !RunBlock(k)) return false;
    }
    memcpy(qpos_out, envt["mujoco_qpos"].data.data(), 64 * 36 * 4);
    npf_out = (int32_t)asFloat(envt["num_pred_frames"], 0);
    return true;
  }
  bool RunHost(int k) {
    std::vector<Ort::Value> iv; std::vector<const char*> inames, onames;
    for (auto& n : Hin[k]) {
      auto it = envt.find(n); if (it == envt.end()) { std::cerr << "✗ [qnn-shim] missing " << n << std::endl; return false; }
      T& t = it->second;
      iv.push_back(Ort::Value::CreateTensor(mi, t.data.data(), t.data.size(), t.shape.data(), t.shape.size(), t.dt));
      inames.push_back(n.c_str());
    }
    for (auto& n : Hout[k]) onames.push_back(n.c_str());
    auto ov = H[k]->Run(Ort::RunOptions{nullptr}, inames.data(), iv.data(), iv.size(), onames.data(), onames.size());
    for (size_t i = 0; i < ov.size(); ++i) {
      auto info = ov[i].GetTensorTypeAndShapeInfo();
      T t; t.dt = info.GetElementType(); t.shape = info.GetShape(); size_t ne = info.GetElementCount();
      t.data.resize(ne * esize(t.dt)); memcpy(t.data.data(), ov[i].GetTensorRawData(), t.data.size());
      envt[Hout[k][i]] = std::move(t);
    }
    return true;
  }
  bool RunBlock(int k) {
    Graph& g = G[k];
    for (uint32_t i = 0; i < g.nIn(); ++i) {
      const std::string& n = Mins[k][inMap[k][i]]; auto it = envt.find(n); if (it == envt.end()) return false;
      size_t ne = telems(g.inputs[i]); float* dst = fbuf(g.inputs[i]);
      if (it->second.data.size() / esize(it->second.dt) != ne) return false;
      if (it->second.dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) memcpy(dst, it->second.data.data(), ne * 4);
      else for (size_t j = 0; j < ne; ++j) dst[j] = asFloat(it->second, j);
    }
    if (!Runtime::Get().Exec(g)) { std::cerr << "✗ [qnn-shim] exec " << g.name << std::endl; return false; }
    for (uint32_t i = 0; i < g.nOut(); ++i) {
      uint32_t r = QNN_TENSOR_GET_RANK(g.outputs[i]); uint32_t* d = QNN_TENSOR_GET_DIMENSIONS(g.outputs[i]);
      setF(Mouts[k][outMap[k][i]], std::vector<int64_t>(d, d + r), fbuf(g.outputs[i]));
    }
    return true;
  }
};

}  // namespace qshim
