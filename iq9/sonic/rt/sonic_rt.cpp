//==============================================================================
// sonic_rt.cpp — IQ-9075 runtime for the GEAR-SONIC encoder/decoder + split planner (groot-virtual-robot-bench; not upstream).
//
// One process, one QNN HTP backend, up to two HTP devices (NSP0 = deviceId 0, NSP1 = deviceId 1),
// each pinned STRICTLY via QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO (unknown id = hard error).
//
//  * EncDec   : SONIC encoder-core + decoder context bins (encdec/), host FSQ round/16 + mode select.
//  * Planner  : split planner = 5 host onnxruntime stages (H0..H4, all ArgMax/TopK/OneHot/CumSum/
//               ScatterND/seed-hash ops, fp32) interleaved with 4 QNN fp16 blocks S,R,P,D.
//
// Modes:
//  --mode planner_correct : run every logged planner call once (in order); write qpos/npf/P-logits
//                           and (optionally, --dump_blocks) every block's inputs/outputs; per-call timing.
//  --mode loaded          : control thread = enc/dec paced at --period_us (50 Hz) for --reps ticks
//                           (same loop as sonic_bench latency mode); optional planner thread
//                           paced at --planner_period_us cycling through the logged calls.
//
// Never SIGKILL this process while it runs on the DSP. It exits by itself (or on SIGINT/SIGTERM at
// the next tick boundary).
//==============================================================================
#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
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

#include "onnxruntime_cxx_api.h"
#include "rtctl.hpp"  // thread naming + env scheduling knobs (SONIC_CTRL_*, SONIC_PLAN_*, SONIC_ORT_*)

using namespace qnn;
using namespace qnn::tools;

namespace {
std::atomic<bool> g_stop{false};
void onSig(int) { g_stop = true; }

uint64_t nowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

sample_app::QnnFunctionPointers g_fp;
void *g_backendHandle = nullptr, *g_modelHandle = nullptr;
Qnn_LogHandle_t g_logHandle = nullptr;
Qnn_BackendHandle_t g_backend = nullptr;
std::map<int, Qnn_DeviceHandle_t> g_devices;
iotensor::IOTensor g_io;

struct Graph {
  std::string name;
  int device = -1;
  Qnn_ContextHandle_t context = nullptr;
  qnn_wrapper_api::GraphInfo_t **graphsInfo = nullptr;
  uint32_t graphsCount = 0;
  Qnn_Tensor_t *inputs = nullptr, *outputs = nullptr;
  uint32_t nIn() const { return (*graphsInfo)[0].numInputTensors; }
  uint32_t nOut() const { return (*graphsInfo)[0].numOutputTensors; }
};
inline float *fbuf(Qnn_Tensor_t &t) { return static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(t).data); }
inline size_t telems(Qnn_Tensor_t &t) {
  uint32_t r = QNN_TENSOR_GET_RANK(t); uint32_t *d = QNN_TENSOR_GET_DIMENSIONS(t);
  size_t n = 1; for (uint32_t i = 0; i < r; ++i) n *= d[i]; return n;
}
inline std::string tname(Qnn_Tensor_t &t) { return QNN_TENSOR_GET_NAME(t); }

// Strict device pinning: the hwDevices[] entry whose deviceId == id, or a hard error.
bool createDevice(int id) {
  if (g_devices.count(id)) return true;
  const QnnDevice_PlatformInfo_t *pi = nullptr;
  if (!g_fp.qnnInterface.deviceGetPlatformInfo || g_fp.qnnInterface.deviceGetPlatformInfo(g_logHandle, &pi) != QNN_SUCCESS || !pi) {
    std::cerr << "ERR deviceGetPlatformInfo\n"; return false; }
  const QnnDevice_HardwareDeviceInfo_t *hit = nullptr;
  std::cerr << "[DEV] platform reports " << pi->v1.numHwDevices << " hw devices:";
  for (uint32_t i = 0; i < pi->v1.numHwDevices; ++i) {
    std::cerr << " id=" << pi->v1.hwDevices[i].v1.deviceId << "(cores=" << pi->v1.hwDevices[i].v1.numCores << ")";
    if ((int)pi->v1.hwDevices[i].v1.deviceId == id) hit = &pi->v1.hwDevices[i];
  }
  std::cerr << "\n";
  if (!hit) { std::cerr << "ERR no HTP device with deviceId " << id << " (strict: no fallback)\n"; return false; }
  static QnnDevice_HardwareDeviceInfo_t sel[4]; static QnnDevice_PlatformInfo_t spi[4]; static QnnDevice_Config_t cfg[4];
  static const QnnDevice_Config_t *cfgp[4][2];
  int k = (int)g_devices.size();
  sel[k] = *hit;
  spi[k].version = QNN_DEVICE_PLATFORM_INFO_VERSION_1; spi[k].v1.numHwDevices = 1; spi[k].v1.hwDevices = &sel[k];
  cfg[k].option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO; cfg[k].hardwareInfo = &spi[k];
  cfgp[k][0] = &cfg[k]; cfgp[k][1] = nullptr;
  Qnn_DeviceHandle_t h = nullptr;
  if (g_fp.qnnInterface.deviceCreate(g_logHandle, cfgp[k], &h) != QNN_SUCCESS) { std::cerr << "ERR deviceCreate id " << id << "\n"; return false; }
  g_devices[id] = h;
  std::cerr << "[DEV] created device handle for deviceId " << id << "\n";
  return true;
}

bool loadGraph(const std::string &path, int dev, Graph &G) {
  if (!createDevice(dev)) return false;
  tools::datautil::StatusCode ds; uint64_t sz = 0;
  std::tie(ds, sz) = tools::datautil::getFileSize(path);
  if (sz == 0) { std::cerr << "ERR empty " << path << "\n"; return false; }
  std::vector<uint8_t> buf(sz);
  if (tools::datautil::readBinaryFromFile(path, buf.data(), sz) != tools::datautil::StatusCode::SUCCESS) return false;
  QnnSystemContext_Handle_t sys = nullptr;
  if (g_fp.qnnSystemInterface.systemContextCreate(&sys) != QNN_SUCCESS) return false;
  const QnnSystemContext_BinaryInfo_t *bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
  if (g_fp.qnnSystemInterface.systemContextGetBinaryInfo(sys, buf.data(), sz, &bi, &bis) != QNN_SUCCESS) return false;
  if (!sample_app::copyMetadataToGraphsInfo(bi, G.graphsInfo, G.graphsCount)) return false;
  g_fp.qnnSystemInterface.systemContextFree(sys);
  if (g_fp.qnnInterface.contextCreateFromBinary(g_backend, g_devices[dev], nullptr, buf.data(), sz, &G.context, nullptr) != QNN_SUCCESS) {
    std::cerr << "ERR contextCreateFromBinary " << path << "\n"; return false; }
  if (g_fp.qnnInterface.graphRetrieve(G.context, (*G.graphsInfo)[0].graphName, &((*G.graphsInfo)[0].graph)) != QNN_SUCCESS) return false;
  if (g_io.setupInputAndOutputTensors(&G.inputs, &G.outputs, (*G.graphsInfo)[0]) != iotensor::StatusCode::SUCCESS) return false;
  G.name = (*G.graphsInfo)[0].graphName; G.device = dev;
  for (uint32_t i = 0; i < G.nIn(); ++i)
    if (QNN_TENSOR_GET_DATA_TYPE(G.inputs[i]) != QNN_DATATYPE_FLOAT_32) { std::cerr << "ERR non-float input " << tname(G.inputs[i]) << "\n"; return false; }
  for (uint32_t i = 0; i < G.nOut(); ++i)
    if (QNN_TENSOR_GET_DATA_TYPE(G.outputs[i]) != QNN_DATATYPE_FLOAT_32) { std::cerr << "ERR non-float output " << tname(G.outputs[i]) << "\n"; return false; }
  std::cerr << "[LOAD] " << G.name << " on deviceId " << dev << " (" << sz / 1048576 << " MB)\n";
  return true;
}
bool exec(Graph &G) {
  return g_fp.qnnInterface.graphExecute((*G.graphsInfo)[0].graph, G.inputs, G.nIn(), G.outputs, G.nOut(), nullptr, nullptr) == QNN_GRAPH_NO_ERROR;
}
void freeGraph(Graph &G) {
  if (!G.context) return;
  g_io.tearDownInputAndOutputTensors(G.inputs, G.outputs, G.nIn(), G.nOut());
  g_fp.qnnInterface.contextFree(G.context, nullptr); G.context = nullptr;
}

int setPerf(int dev, const std::string &mode) {
  if (mode == "none") return 0;
  QnnDevice_Infrastructure_t inf = nullptr;
  if (!g_fp.qnnInterface.deviceGetInfrastructure || g_fp.qnnInterface.deviceGetInfrastructure(&inf) != QNN_SUCCESS) { std::cerr << "WARN no device infrastructure\n"; return 1; }
  auto &pi = static_cast<QnnHtpDevice_Infrastructure_t *>(inf)->perfInfra;
  uint32_t id = 0;
  if (pi.createPowerConfigId((uint32_t)dev, 0, &id) != QNN_SUCCESS) { std::cerr << "WARN createPowerConfigId dev " << dev << "\n"; return 1; }
  QnnHtpPerfInfrastructure_PowerConfig_t dc; memset(&dc, 0, sizeof(dc));
  dc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto &v = dc.dcvsV3Config;
  v.contextId = id; v.setDcvsEnable = 1; v.dcvsEnable = 0;
  v.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  v.setSleepLatency = 1; v.sleepLatency = 40; v.setSleepDisable = 1; v.sleepDisable = 1;
  v.setBusParams = 1; v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  v.setCoreParams = 1; v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = {&dc, nullptr};
  Qnn_ErrorHandle_t e = pi.setPowerConfig(id, cfgs);
  std::cerr << "[PERF] device " << dev << " mode=" << mode << " powerConfigId=" << id << " setPowerConfig rc=" << e << "\n";
  return e == QNN_SUCCESS ? 0 : 1;
}

bool readAll(const std::string &p, std::vector<float> &v) {
  FILE *f = fopen(p.c_str(), "rb"); if (!f) return false;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  v.resize(n / 4); size_t g = fread(v.data(), 4, v.size(), f); fclose(f); return g == v.size();
}
void writeAll(const std::string &p, const void *d, size_t bytes) {
  FILE *f = fopen(p.c_str(), "wb"); if (!f) return; fwrite(d, 1, bytes, f); fclose(f);
}
std::string sanitize(const std::string &s) {
  std::string o = s; for (auto &c : o) if (!isalnum((unsigned char)c) && c != '_') c = '_';
  return o;
}

//------------------------------------------------------------------ planner
struct T { ONNXTensorElementDataType dt; std::vector<int64_t> shape; std::vector<uint8_t> data; };
size_t esize(ONNXTensorElementDataType dt) {
  switch (dt) { case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
    default: return 0; }
}
float asFloat(const T &t, size_t i) {
  switch (t.dt) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return ((const float *)t.data.data())[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return (float)((const int64_t *)t.data.data())[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return (float)((const int32_t *)t.data.data())[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return ((const uint8_t *)t.data.data())[i] ? 1.f : 0.f;
    default: return NAN;
  }
}
struct BlockMap { std::string block; std::vector<std::string> ins, outs; };
struct Planner {
  Ort::Env &env;
  std::vector<std::unique_ptr<Ort::Session>> H;
  std::vector<std::vector<std::string>> Hin, Hout;
  Graph G[4]; BlockMap M[4];
  std::vector<std::vector<int>> inMap, outMap;  // per block: qnn input i -> index into M.ins ; qnn output -> M.outs
  std::map<std::string, T> envt;
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  double tStage[9];  // H0 S H1 R H2 P H3 D H4 (us)
  bool dump = false; std::vector<std::vector<float>> dumpIn[4], dumpOut[4];
  explicit Planner(Ort::Env &e) : env(e) {}
  bool load(const std::string &dir, int dev, int threads) {
    Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetInterOpNumThreads(1);
    so.SetCustomCreateThreadFn(rtctl::ort_create); so.SetCustomJoinThreadFn(rtctl::ort_join);
    { const char* sp = std::getenv("SONIC_ORT_SPIN"); if (sp && std::string(sp) == "0") so.AddConfigEntry("session.intra_op.allow_spinning", "0");
      std::cerr << "[rtctl] planner ORT intra_op_threads=" << threads << " allow_spinning=" << (sp ? sp : "1(default)") << std::endl; }
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    for (int k = 0; k < 5; ++k) {
      std::string p = dir + "/host_H" + std::to_string(k) + ".onnx";
      H.emplace_back(new Ort::Session(env, p.c_str(), so));
      Ort::AllocatorWithDefaultOptions al;
      std::vector<std::string> in, out;
      for (size_t i = 0; i < H[k]->GetInputCount(); ++i) in.push_back(H[k]->GetInputNameAllocated(i, al).get());
      for (size_t i = 0; i < H[k]->GetOutputCount(); ++i) out.push_back(H[k]->GetOutputNameAllocated(i, al).get());
      Hin.push_back(in); Hout.push_back(out);
      std::cerr << "[ORT] H" << k << " in=" << in.size() << " out=" << out.size() << "\n";
    }
    // manifest: "block X" / "in name" / "out name"
    std::ifstream mf(dir + "/manifest.txt"); std::string a, b; int cur = -1;
    while (mf >> a >> b) {
      if (a == "block") { cur++; M[cur].block = b; }
      else if (a == "in") M[cur].ins.push_back(b); else if (a == "out") M[cur].outs.push_back(b);
    }
    if (cur != 3) { std::cerr << "ERR manifest\n"; return false; }
    const char *bn[4] = {"S", "R", "P", "D"};
    inMap.resize(4); outMap.resize(4);
    for (int k = 0; k < 4; ++k) {
      if (!loadGraph(dir + "/planner_" + bn[k] + "_aaf0_fp16.bin", dev, G[k])) return false;
      for (uint32_t i = 0; i < G[k].nIn(); ++i) {
        std::string q = tname(G[k].inputs[i]); int hit = -1;
        for (size_t j = 0; j < M[k].ins.size(); ++j)
          if (sanitize(M[k].ins[j]) == q || sanitize(M[k].ins[j] + "__f32") == q) hit = (int)j;
        if (hit < 0) { std::cerr << "ERR unmapped qnn input " << q << "\n"; return false; }
        inMap[k].push_back(hit);
      }
      for (uint32_t i = 0; i < G[k].nOut(); ++i) {
        std::string q = tname(G[k].outputs[i]); int hit = -1;
        for (size_t j = 0; j < M[k].outs.size(); ++j) if (sanitize(M[k].outs[j]) == q) hit = (int)j;
        if (hit < 0) { std::cerr << "ERR unmapped qnn output " << q << "\n"; return false; }
        outMap[k].push_back(hit);
      }
    }
    return true;
  }
  void setF(const std::string &n, std::vector<int64_t> sh, const float *v) {
    T t; t.dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; t.shape = sh; size_t ne = 1; for (auto s : sh) ne *= s;
    t.data.resize(ne * 4); memcpy(t.data.data(), v, ne * 4); envt[n] = std::move(t);
  }
  void setI64(const std::string &n, std::vector<int64_t> sh, std::vector<int64_t> v) {
    T t; t.dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; t.shape = sh; t.data.resize(v.size() * 8); memcpy(t.data.data(), v.data(), v.size() * 8); envt[n] = std::move(t);
  }
  // call record: ctx[144] mode tv mv[3] face[3] seed height  (154 floats)
  void setInputs(const float *c) {
    envt.clear();
    setF("context_mujoco_qpos", {1, 4, 36}, c);
    setI64("mode", {1}, {(int64_t)c[144]});
    setF("target_vel", {1}, c + 145);
    setF("movement_direction", {1, 3}, c + 146);
    setF("facing_direction", {1, 3}, c + 149);
    setI64("random_seed", {1}, {(int64_t)c[152]});
    setF("height", {1}, c + 153);
    setI64("has_specific_target", {1, 1}, {0});
    float z[12] = {0}; setF("specific_target_positions", {1, 4, 3}, z); setF("specific_target_headings", {1, 4}, z);
    setI64("allowed_pred_num_tokens", {1, 11}, {0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0});
  }
  bool runHost(int k) {
    std::vector<Ort::Value> iv; std::vector<const char *> inames, onames;
    for (auto &n : Hin[k]) {
      auto it = envt.find(n); if (it == envt.end()) { std::cerr << "ERR missing host input " << n << "\n"; return false; }
      T &t = it->second;
      iv.push_back(Ort::Value::CreateTensor(mi, t.data.data(), t.data.size(), t.shape.data(), t.shape.size(), t.dt));
      inames.push_back(n.c_str());
    }
    for (auto &n : Hout[k]) onames.push_back(n.c_str());
    auto ov = H[k]->Run(Ort::RunOptions{nullptr}, inames.data(), iv.data(), iv.size(), onames.data(), onames.size());
    for (size_t i = 0; i < ov.size(); ++i) {
      auto info = ov[i].GetTensorTypeAndShapeInfo();
      T t; t.dt = info.GetElementType(); t.shape = info.GetShape(); size_t ne = info.GetElementCount();
      t.data.resize(ne * esize(t.dt)); memcpy(t.data.data(), ov[i].GetTensorRawData(), t.data.size());
      envt[Hout[k][i]] = std::move(t);
    }
    return true;
  }
  bool runBlock(int k) {
    Graph &g = G[k];
    for (uint32_t i = 0; i < g.nIn(); ++i) {
      const std::string &n = M[k].ins[inMap[k][i]];
      auto it = envt.find(n); if (it == envt.end()) { std::cerr << "ERR missing block input " << n << "\n"; return false; }
      size_t ne = telems(g.inputs[i]); float *dst = fbuf(g.inputs[i]);
      if (it->second.data.size() / esize(it->second.dt) != ne) { std::cerr << "ERR size " << n << "\n"; return false; }
      if (it->second.dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) memcpy(dst, it->second.data.data(), ne * 4);
      else for (size_t j = 0; j < ne; ++j) dst[j] = asFloat(it->second, j);
      if (dump) dumpIn[k].emplace_back(dst, dst + ne);
    }
    if (!exec(g)) { std::cerr << "ERR exec " << g.name << "\n"; return false; }
    for (uint32_t i = 0; i < g.nOut(); ++i) {
      const std::string &n = M[k].outs[outMap[k][i]];
      uint32_t r = QNN_TENSOR_GET_RANK(g.outputs[i]); uint32_t *d = QNN_TENSOR_GET_DIMENSIONS(g.outputs[i]);
      std::vector<int64_t> sh(d, d + r);
      setF(n, sh, fbuf(g.outputs[i]));
      if (dump) { size_t ne = telems(g.outputs[i]); dumpOut[k].emplace_back(fbuf(g.outputs[i]), fbuf(g.outputs[i]) + ne); }
    }
    return true;
  }
  bool run(const float *call) {
    setInputs(call);
    for (int k = 0; k < 5; ++k) {
      uint64_t t0 = nowUs(); if (!runHost(k)) return false; tStage[2 * k] = (double)(nowUs() - t0);
      if (k < 4) { t0 = nowUs(); if (!runBlock(k)) return false; tStage[2 * k + 1] = (double)(nowUs() - t0); }
    }
    return true;
  }
  const float *qpos() { return (const float *)envt["mujoco_qpos"].data.data(); }
  int npf() { return (int)asFloat(envt["num_pred_frames"], 0); }
  const float *plogits() { return (const float *)envt["/mmm_net/_proj_pose_output_logit/Add_output_0"].data.data(); }
};

//------------------------------------------------------------------ enc/dec (same loop as sonic_bench)
struct EncDec {
  Graph E, D; std::vector<float> ein, din; size_t EI = 0, EO = 0, DI = 0, DO = 0; long N = 0;
  bool load(const std::string &enc, const std::string &dec, int dev, const std::string &encIn, const std::string &decIn) {
    if (!loadGraph(enc, dev, E) || !loadGraph(dec, dev, D)) return false;
    if (!readAll(encIn, ein) || !readAll(decIn, din)) { std::cerr << "ERR read enc/dec inputs\n"; return false; }
    EI = telems(E.inputs[0]); EO = telems(E.outputs[0]); DI = telems(D.inputs[0]); DO = telems(D.outputs[0]);
    N = (long)(ein.size() / EI);
    if ((long)(din.size() / DI) != N || EO != 192) { std::cerr << "ERR enc/dec shapes\n"; return false; }
    return true;
  }
  // one control step; returns false on exec error. times in us
  bool step(long i, uint64_t &e_us, uint64_t &d_us, float *act) {
    memcpy(fbuf(E.inputs[0]), &ein[(size_t)i * EI], EI * 4);
    uint64_t a0 = nowUs(); if (!exec(E)) return false; uint64_t a1 = nowUs();
    float *dinb = fbuf(D.inputs[0]); memcpy(dinb, &din[(size_t)i * DI], DI * 4);
    const float *eo = fbuf(E.outputs[0]);
    int k = (int)ein[(size_t)i * EI + 0]; if (k < 0 || k > 2) k = 0;
    for (int j = 0; j < 64; ++j) dinb[j] = std::nearbyint(eo[k * 64 + j]) / 16.0f;
    uint64_t b0 = nowUs(); if (!exec(D)) return false; uint64_t b1 = nowUs();
    if (act) memcpy(act, fbuf(D.outputs[0]), DO * 4);
    e_us = a1 - a0; d_us = b1 - b0; return true;
  }
};
}  // namespace

int main(int argc, char **argv) {
  std::string backend, syslib, mode = "planner_correct", out = ".", pdir, calls, enc, dec, encIn, decIn, perf = "burst";
  int pdev = 1, edev = 0, pthreads = 4; long reps = 10000, warm = 100, period = 20000, pperiod = 100000, plimit = 0;
  bool dumpBlocks = false, withPlanner = false; std::string plant;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
    if (a == "--backend") backend = nx(); else if (a == "--system_library") syslib = nx();
    else if (a == "--mode") mode = nx(); else if (a == "--out") out = nx();
    else if (a == "--planner_dir") pdir = nx(); else if (a == "--calls") calls = nx();
    else if (a == "--planner_device") pdev = std::stoi(nx()); else if (a == "--encdec_device") edev = std::stoi(nx());
    else if (a == "--planner_threads") pthreads = std::stoi(nx());
    else if (a == "--enc") enc = nx(); else if (a == "--dec") dec = nx();
    else if (a == "--enc_in") encIn = nx(); else if (a == "--dec_in") decIn = nx();
    else if (a == "--perf") perf = nx();
    else if (a == "--reps") reps = std::stol(nx()); else if (a == "--warmup") warm = std::stol(nx());
    else if (a == "--period_us") period = std::stol(nx()); else if (a == "--planner_period_us") pperiod = std::stol(nx());
    else if (a == "--planner") withPlanner = true; else if (a == "--dump_blocks") dumpBlocks = true;
    else if (a == "--n") plimit = std::stol(nx()); else if (a == "--plant") plant = nx();
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }
  signal(SIGINT, onSig); signal(SIGTERM, onSig);
  qnn::log::initializeLogging(); qnn::log::setLogLevel(QNN_LOG_LEVEL_ERROR);
  if (dynamicloadutil::getQnnFunctionPointers(backend, "", &g_fp, &g_backendHandle, false, &g_modelHandle) != dynamicloadutil::StatusCode::SUCCESS) { std::cerr << "ERR backend\n"; return 1; }
  if (dynamicloadutil::getQnnSystemFunctionPointers(syslib, &g_fp) != dynamicloadutil::StatusCode::SUCCESS) { std::cerr << "ERR syslib\n"; return 1; }
  if (g_fp.qnnInterface.logCreate) g_fp.qnnInterface.logCreate(qnn::log::getLogCallback(), qnn::log::getLogLevel(), &g_logHandle);
  if (g_fp.qnnInterface.backendCreate(g_logHandle, nullptr, &g_backend) != QNN_BACKEND_NO_ERROR) { std::cerr << "ERR backendCreate\n"; return 1; }
  Ort::Env oenv(ORT_LOGGING_LEVEL_WARNING, "sonic_rt");
  int rc = 0;
  std::vector<float> C; long NC = 0;
  if (!calls.empty()) { if (!readAll(calls, C)) { std::cerr << "ERR calls\n"; return 1; } NC = (long)(C.size() / 154); if (plimit > 0 && plimit < NC) NC = plimit; }

  if (mode == "planner_correct") {
    Planner P(oenv); P.dump = dumpBlocks;
    uint64_t t0 = nowUs();
    if (!P.load(pdir, pdev, pthreads)) return 1;
    std::cerr << "[LOAD] planner us=" << nowUs() - t0 << "\n";
    setPerf(pdev, perf);
    // plant: reverse the feature axis of the P block's /Add_135 input (planted fault: the gate must fail)
    std::vector<float> qp((size_t)NC * 64 * 36), lg((size_t)NC * 16 * 80); std::vector<int32_t> nf(NC);
    FILE *csv = fopen((out + "/planner_calls.csv").c_str(), "w");
    fprintf(csv, "call,total_us,H0,S,H1,R,H2,P,H3,D,H4,npf\n");
    for (long i = 0; i < NC && !g_stop; ++i) {
      if (!plant.empty()) {  // run the stages manually with the fault injected before block P
        P.setInputs(&C[(size_t)i * 154]);
        for (int k = 0; k < 5; ++k) {
          if (!P.runHost(k)) return 1;
          if (k == 2) { T &t = P.envt["/Add_135_output_0"]; float *f = (float *)t.data.data(); size_t ne = t.data.size() / 4, w = (size_t)t.shape.back();
            for (size_t r = 0; r < ne / w; ++r) std::reverse(f + r * w, f + r * w + w); }
          if (k < 4 && !P.runBlock(k)) return 1;
        }
        for (double &x : P.tStage) x = 0;
      } else if (!P.run(&C[(size_t)i * 154])) { rc = 1; break; }
      double tot = 0; for (double x : P.tStage) tot += x;
      memcpy(&qp[(size_t)i * 2304], P.qpos(), 2304 * 4); memcpy(&lg[(size_t)i * 1280], P.plogits(), 1280 * 4); nf[i] = P.npf();
      fprintf(csv, "%ld,%.0f", i, tot); for (double x : P.tStage) fprintf(csv, ",%.0f", x); fprintf(csv, ",%d\n", nf[i]);
    }
    fclose(csv);
    writeAll(out + "/qpos.f32", qp.data(), qp.size() * 4); writeAll(out + "/plogits.f32", lg.data(), lg.size() * 4); writeAll(out + "/npf.i32", nf.data(), nf.size() * 4);
    if (dumpBlocks) for (int k = 0; k < 4; ++k) {
      std::vector<float> a, b; for (auto &v : P.dumpIn[k]) a.insert(a.end(), v.begin(), v.end()); for (auto &v : P.dumpOut[k]) b.insert(b.end(), v.begin(), v.end());
      writeAll(out + "/blk" + std::to_string(k) + "_in.f32", a.data(), a.size() * 4); writeAll(out + "/blk" + std::to_string(k) + "_out.f32", b.data(), b.size() * 4);
    }
    for (auto &g : P.G) freeGraph(g);
    std::cout << "PLANNER_CORRECT_DONE calls=" << NC << " device=" << pdev << std::endl;
  } else if (mode == "loaded") {
    EncDec ED;
    if (!ED.load(enc, dec, edev, encIn, decIn)) return 1;
    std::unique_ptr<Planner> P;
    if (withPlanner) { P.reset(new Planner(oenv)); if (!P->load(pdir, pdev, pthreads)) return 1; }
    setPerf(edev, perf); if (withPlanner && pdev != edev) setPerf(pdev, perf);
    std::atomic<bool> ctlDone{false}; std::atomic<long> plannerCalls{0};
    std::vector<std::array<uint64_t, 6>> prec;  // per planner call: t_start, total, late, idx, npf, ok
    double stageSum[9] = {0};
    std::thread pt;
    std::atomic<bool> go{false};
    if (withPlanner) {
      pt = std::thread([&]() {
        rtctl::apply("SONIC_PLAN", "sonic_plan");
        while (!go) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        uint64_t next = nowUs(); long i = 0;
        while (!ctlDone && !g_stop) {
          uint64_t n0 = nowUs(), late = 0;
          if (n0 < next) std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::microseconds(next)));
          else late = n0 - next;
          next += pperiod;
          if (ctlDone) break;
          uint64_t s0 = nowUs(); bool ok = P->run(&C[(size_t)(i % NC) * 154]); uint64_t s1 = nowUs();
          for (int k = 0; k < 9; ++k) stageSum[k] += P->tStage[k];
          prec.push_back({s0, s1 - s0, late, (uint64_t)(i % NC), (uint64_t)(ok ? P->npf() : 0), (uint64_t)ok});
          if (!ok) break;
          ++i; plannerCalls = i;
          if (s1 > next) next = s1;  // a call longer than the period skips the missed slots (upstream scheduler semantics)
        }
      });
    }
    FILE *csv = fopen((out + "/lat.csv").c_str(), "w");
    fprintf(csv, "rep,i,t_start_us,enc_us,dec_us,glue_us,step_us,late_us\n");
    std::vector<float> acts((size_t)reps * ED.DO);
    rtctl::apply("SONIC_CTRL", "sonic_ctrl");  // control loop = main thread (after the planner thread exists)
    uint64_t next = nowUs(); go = true;
    for (long r = -warm; r < reps && !g_stop; ++r) {
      long i = ((r % ED.N) + ED.N) % ED.N; uint64_t late = 0;
      if (period > 0) {
        uint64_t n0 = nowUs();
        if (n0 < next) std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::microseconds(next)));
        else late = n0 - next;
        next += period;
      }
      uint64_t s0 = nowUs(), e = 0, d = 0;
      if (!ED.step(i, e, d, r >= 0 ? &acts[(size_t)r * ED.DO] : nullptr)) { std::cerr << "ERR encdec\n"; rc = 1; break; }
      uint64_t s1 = nowUs();
      if (r >= 0) fprintf(csv, "%ld,%ld,%lu,%lu,%lu,%lu,%lu,%lu\n", r, i, (unsigned long)s0, (unsigned long)e, (unsigned long)d,
                          (unsigned long)((s1 - s0) - e - d), (unsigned long)(s1 - s0), (unsigned long)late);
    }
    ctlDone = true; fclose(csv);
    if (pt.joinable()) pt.join();
    writeAll(out + "/lat_actions.f32", acts.data(), acts.size() * 4);
    if (withPlanner) {
      FILE *pc = fopen((out + "/planner_calls.csv").c_str(), "w");
      fprintf(pc, "call,t_start_us,total_us,late_us,idx,npf,ok\n");
      for (size_t j = 0; j < prec.size(); ++j) fprintf(pc, "%zu,%lu,%lu,%lu,%lu,%lu,%lu\n", j, (unsigned long)prec[j][0], (unsigned long)prec[j][1], (unsigned long)prec[j][2], (unsigned long)prec[j][3], (unsigned long)prec[j][4], (unsigned long)prec[j][5]);
      fclose(pc);
      FILE *ps = fopen((out + "/planner_stage_mean_us.txt").c_str(), "w");
      const char *sn[9] = {"H0", "S", "H1", "R", "H2", "P", "H3", "D", "H4"};
      for (int k = 0; k < 9; ++k) fprintf(ps, "%s %.1f\n", sn[k], prec.empty() ? 0 : stageSum[k] / prec.size());
      fclose(ps);
      for (auto &g : P->G) freeGraph(g);
    }
    freeGraph(ED.E); freeGraph(ED.D);
    std::cout << "LOADED_DONE reps=" << reps << " planner=" << withPlanner << " pdev=" << pdev << " edev=" << edev
              << " planner_calls=" << plannerCalls << " stopped=" << g_stop << std::endl;
  } else { std::cerr << "unknown mode\n"; rc = 2; }
  for (auto &kv : g_devices) if (g_fp.qnnInterface.deviceFree) g_fp.qnnInterface.deviceFree(kv.second);
  if (g_backend && g_fp.qnnInterface.backendFree) g_fp.qnnInterface.backendFree(g_backend);
  return rc;
}
