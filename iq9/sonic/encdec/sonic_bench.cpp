//==============================================================================
// sonic_bench.cpp — resident QNN runner for the GEAR-SONIC encoder+decoder on the IQ-9075 (Hexagon v73).
// groot-virtual-robot-bench; not part of upstream. Build: iq9/sonic/encdec/build_bench.sh (needs the QAIRT SDK's SampleApp sources).
//
// Loads the encoder and decoder context binaries ONCE on one backend+device (one NSP), sets up
// float32 I/O tensors ONCE, then runs ticks entirely in memory (no .raw files, no per-call
// reload). Two modes:
//
//  --mode correct : for each of N ticks, enc(enc_in[i]) and dec(dec_in[i]) — decoder fed the LOGGED
//                   dec_in so its error is isolated. Writes enc_raw.f32 [N, enc_out_elems] and
//                   dec_raw.f32 [N, 29] for the host gate.
//  --mode latency : control-step loop, rep r uses tick i=r%N:
//                   copy enc_in[i] -> enc.execute -> host FSQ(Round/16)+mode-select (core graph)
//                   -> write 64 tokens into dec_in[i][0:64] -> dec.execute -> copy action out.
//                   Per rep it records (us): enc_exec, dec_exec (graphExecute wall = ARM->DSP fastRPC
//                   round trip + DSP execute), glue (host FSQ/select/copies), step (whole tick).
//                   --period_us P paces ticks (e.g. 20000 = 50 Hz) so the DSP sees the real duty
//                   cycle (sleep/DCVS between ticks), 0 = back-to-back.
//                   Writes lat.csv (rep,i,enc_us,dec_us,glue_us,step_us) and lat_actions.f32.
//
// --profile : QNN basic profiling per call -> enc_acc_us/dec_acc_us = DSP accelerator execute time
//             (event 3004). Profiling adds overhead: use the NON-profiled run for wall/step numbers.
// Perf vote (--perf): none | burst (DCVS_V3 off, perf mode, max corners, sleep disabled)
//                     | burst_poll (burst + RPC polling 9999us + RPC control latency 100us).
//
// late_us only counts a tick that was already late BEFORE its pacing sleep; a wake-up delay after the sleep is not in it.
// To measure start lateness use start-to-start gaps (rt/sonic_rt writes t_start_us; rt/gaps.py).
//
// Never SIGKILL this process while running on the DSP. It exits on its own after the reps.
//==============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
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
#include "QnnProfile.h"

using namespace qnn;
using namespace qnn::tools;

namespace {
struct Loaded {
  std::string name;
  Qnn_ContextHandle_t context = nullptr;
  qnn_wrapper_api::GraphInfo_t **graphsInfo = nullptr;
  uint32_t graphsCount = 0;
  Qnn_Tensor_t *inputs = nullptr;
  Qnn_Tensor_t *outputs = nullptr;
};
sample_app::QnnFunctionPointers g_fp;
void *g_backendHandle = nullptr, *g_modelHandle = nullptr;
Qnn_LogHandle_t g_logHandle = nullptr;
Qnn_BackendHandle_t g_backend = nullptr;
Qnn_DeviceHandle_t g_device = nullptr;
iotensor::IOTensor g_io;

uint64_t nowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline float *fbuf(Qnn_Tensor_t &t) { return static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(t).data); }
inline size_t telems(Qnn_Tensor_t &t) {
  uint32_t r = QNN_TENSOR_GET_RANK(t); uint32_t *d = QNN_TENSOR_GET_DIMENSIONS(t);
  size_t n = 1; for (uint32_t i = 0; i < r; ++i) n *= d[i]; return n;
}

bool loadContext(const std::string &path, Loaded &L) {
  tools::datautil::StatusCode ds; uint64_t sz = 0;
  std::tie(ds, sz) = tools::datautil::getFileSize(path);
  if (sz == 0) { std::cerr << "ERR empty " << path << "\n"; return false; }
  std::vector<uint8_t> buf(sz);
  if (tools::datautil::readBinaryFromFile(path, buf.data(), sz) != tools::datautil::StatusCode::SUCCESS) return false;
  QnnSystemContext_Handle_t sys = nullptr;
  if (g_fp.qnnSystemInterface.systemContextCreate(&sys) != QNN_SUCCESS) return false;
  const QnnSystemContext_BinaryInfo_t *bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
  if (g_fp.qnnSystemInterface.systemContextGetBinaryInfo(sys, buf.data(), sz, &bi, &bis) != QNN_SUCCESS) return false;
  if (!sample_app::copyMetadataToGraphsInfo(bi, L.graphsInfo, L.graphsCount)) return false;
  g_fp.qnnSystemInterface.systemContextFree(sys);
  if (g_fp.qnnInterface.contextCreateFromBinary(g_backend, g_device, nullptr, buf.data(), sz, &L.context, nullptr) != QNN_SUCCESS) {
    std::cerr << "ERR contextCreateFromBinary " << path << "\n"; return false; }
  if (g_fp.qnnInterface.graphRetrieve(L.context, (*L.graphsInfo)[0].graphName, &((*L.graphsInfo)[0].graph)) != QNN_SUCCESS) return false;
  if (g_io.setupInputAndOutputTensors(&L.inputs, &L.outputs, (*L.graphsInfo)[0]) != iotensor::StatusCode::SUCCESS) return false;
  L.name = (*L.graphsInfo)[0].graphName;
  for (uint32_t i = 0; i < (*L.graphsInfo)[0].numInputTensors; ++i)
    std::cerr << "[IO] " << L.name << " IN " << QNN_TENSOR_GET_NAME(L.inputs[i]) << " dtype=" << (int)QNN_TENSOR_GET_DATA_TYPE(L.inputs[i]) << " elems=" << telems(L.inputs[i]) << "\n";
  for (uint32_t i = 0; i < (*L.graphsInfo)[0].numOutputTensors; ++i)
    std::cerr << "[IO] " << L.name << " OUT " << QNN_TENSOR_GET_NAME(L.outputs[i]) << " dtype=" << (int)QNN_TENSOR_GET_DATA_TYPE(L.outputs[i]) << " elems=" << telems(L.outputs[i]) << "\n";
  return true;
}
bool g_profile = false;
uint64_t g_lastAccUs = 0, g_lastHtpRpcUs = 0;
void scanEvents(const QnnProfile_EventId_t *ev, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    QnnProfile_EventData_t d = QNN_PROFILE_EVENT_DATA_INIT;
    if (g_fp.qnnInterface.profileGetEventData(ev[i], &d) == QNN_SUCCESS) {
      if (d.type == 3004) g_lastAccUs = d.value;          // GRAPH_EXECUTE_ACCEL_TIME_MICROSEC
      if (d.type == 3002) g_lastHtpRpcUs = d.value;       // GRAPH_EXECUTE_HTP_RPC_TIME_MICROSEC
    }
    const QnnProfile_EventId_t *sub = nullptr; uint32_t ns = 0;
    if (g_fp.qnnInterface.profileGetSubEvents(ev[i], &sub, &ns) == QNN_SUCCESS && ns) scanEvents(sub, ns);
  }
}
bool exec(Loaded &L) {
  Qnn_ProfileHandle_t ph = nullptr;
  if (g_profile) g_fp.qnnInterface.profileCreate(g_backend, QNN_PROFILE_LEVEL_BASIC, &ph);
  bool ok = g_fp.qnnInterface.graphExecute((*L.graphsInfo)[0].graph, L.inputs, (*L.graphsInfo)[0].numInputTensors,
                                        L.outputs, (*L.graphsInfo)[0].numOutputTensors, ph, nullptr) == QNN_GRAPH_NO_ERROR;
  if (ph) {
    g_lastAccUs = g_lastHtpRpcUs = 0;
    const QnnProfile_EventId_t *ev = nullptr; uint32_t n = 0;
    if (g_fp.qnnInterface.profileGetEvents(ph, &ev, &n) == QNN_SUCCESS) scanEvents(ev, n);
    g_fp.qnnInterface.profileFree(ph);
  }
  return ok;
}
bool readAll(const std::string &p, std::vector<float> &v) {
  FILE *f = fopen(p.c_str(), "rb"); if (!f) return false;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  v.resize(n / 4); size_t g = fread(v.data(), 4, v.size(), f); fclose(f); return g == v.size();
}
bool writeAll(const std::string &p, const float *d, size_t n) {
  FILE *f = fopen(p.c_str(), "wb"); if (!f) return false; fwrite(d, 4, n, f); fclose(f); return true;
}

int setPerf(const std::string &mode) {
  if (mode == "none") return 0;
  QnnDevice_Infrastructure_t inf = nullptr;
  if (!g_fp.qnnInterface.deviceGetInfrastructure || g_fp.qnnInterface.deviceGetInfrastructure(&inf) != QNN_SUCCESS) {
    std::cerr << "WARN no device infrastructure\n"; return 1; }
  auto *htp = static_cast<QnnHtpDevice_Infrastructure_t *>(inf);
  auto &pi = htp->perfInfra;
  uint32_t id = 0;
  if (pi.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) { std::cerr << "WARN createPowerConfigId\n"; return 1; }
  QnnHtpPerfInfrastructure_PowerConfig_t dc; memset(&dc, 0, sizeof(dc));
  dc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto &v = dc.dcvsV3Config;
  v.contextId = id; v.setDcvsEnable = 1; v.dcvsEnable = 0;
  v.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  v.setSleepLatency = 1; v.sleepLatency = 40; v.setSleepDisable = 1; v.sleepDisable = 1;
  v.setBusParams = 1; v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  v.setCoreParams = 1; v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t *> cfgs = {&dc};
  QnnHtpPerfInfrastructure_PowerConfig_t poll, lat; memset(&poll, 0, sizeof(poll)); memset(&lat, 0, sizeof(lat));
  if (mode == "burst_poll") {
    poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME; poll.rpcPollingTimeConfig = 9999;
    lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY; lat.rpcControlLatencyConfig = 100;
    cfgs.push_back(&poll); cfgs.push_back(&lat);
  }
  cfgs.push_back(nullptr);
  Qnn_ErrorHandle_t e = pi.setPowerConfig(id, cfgs.data());
  std::cerr << "[PERF] mode=" << mode << " setPowerConfig rc=" << e << "\n";
  return e == QNN_SUCCESS ? 0 : 1;
}
}  // namespace

int main(int argc, char **argv) {
  std::string backend, syslib, encBin, decBin, encIn, decIn, outDir = ".", mode = "correct", perf = "burst", encKind = "core";
  long reps = 1000, warm = 50, periodUs = 0, nLimit = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
    if (a == "--backend") backend = nx(); else if (a == "--system_library") syslib = nx();
    else if (a == "--enc") encBin = nx(); else if (a == "--dec") decBin = nx();
    else if (a == "--enc_in") encIn = nx(); else if (a == "--dec_in") decIn = nx();
    else if (a == "--out") outDir = nx(); else if (a == "--mode") mode = nx();
    else if (a == "--perf") perf = nx(); else if (a == "--enc_kind") encKind = nx();
    else if (a == "--reps") reps = std::stol(nx()); else if (a == "--warmup") warm = std::stol(nx());
    else if (a == "--profile") g_profile = true;
    else if (a == "--period_us") periodUs = std::stol(nx()); else if (a == "--n") nLimit = std::stol(nx());
  }
  if (backend.empty() || syslib.empty() || encBin.empty() || decBin.empty() || encIn.empty() || decIn.empty()) {
    std::cerr << "usage: sonic_bench --backend libQnnHtp.so --system_library libQnnSystem.so --enc E.bin --dec D.bin "
                 "--enc_in enc_in.f32 --dec_in dec_in.f32 [--mode correct|latency] [--enc_kind core|full] "
                 "[--perf none|burst|burst_poll] [--reps R --warmup W --period_us P --n N] [--out DIR]\n";
    return 2;
  }
  qnn::log::initializeLogging(); qnn::log::setLogLevel(QNN_LOG_LEVEL_ERROR);
  if (dynamicloadutil::getQnnFunctionPointers(backend, "", &g_fp, &g_backendHandle, false, &g_modelHandle) != dynamicloadutil::StatusCode::SUCCESS) { std::cerr << "ERR backend\n"; return 1; }
  if (dynamicloadutil::getQnnSystemFunctionPointers(syslib, &g_fp) != dynamicloadutil::StatusCode::SUCCESS) { std::cerr << "ERR syslib\n"; return 1; }
  if (g_fp.qnnInterface.logCreate) g_fp.qnnInterface.logCreate(qnn::log::getLogCallback(), qnn::log::getLogLevel(), &g_logHandle);
  if (g_fp.qnnInterface.backendCreate(g_logHandle, nullptr, &g_backend) != QNN_BACKEND_NO_ERROR) { std::cerr << "ERR backendCreate\n"; return 1; }
  if (g_fp.qnnInterface.deviceCreate) {
    Qnn_ErrorHandle_t de = g_fp.qnnInterface.deviceCreate(g_logHandle, nullptr, &g_device);
    if (de != QNN_SUCCESS && de != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) { std::cerr << "ERR deviceCreate\n"; return 1; }
  }
  uint64_t t0 = nowUs();
  Loaded E, D;
  if (!loadContext(encBin, E) || !loadContext(decBin, D)) { std::cerr << "ERR load\n"; return 1; }
  std::cerr << "[LOAD] us=" << nowUs() - t0 << "\n";
  int perfRc = setPerf(perf);

  std::vector<float> ein, din;
  if (!readAll(encIn, ein) || !readAll(decIn, din)) { std::cerr << "ERR read inputs\n"; return 1; }
  const size_t EI = telems(E.inputs[0]), EO = telems(E.outputs[0]), DI = telems(D.inputs[0]), DO = telems(D.outputs[0]);
  long N = (long)(ein.size() / EI);
  if ((long)(din.size() / DI) != N) { std::cerr << "ERR N mismatch\n"; return 1; }
  if (nLimit > 0 && nLimit < N) N = nLimit;
  std::cerr << "[DATA] N=" << N << " EI=" << EI << " EO=" << EO << " DI=" << DI << " DO=" << DO << "\n";
  if (encKind == "core" && EO != 3 * 64) { std::cerr << "ERR core expects 192 outs\n"; return 1; }

  if (mode == "correct") {
    std::vector<float> eo((size_t)N * EO), dout((size_t)N * DO);
    for (long i = 0; i < N; ++i) {
      memcpy(fbuf(E.inputs[0]), &ein[(size_t)i * EI], EI * 4);
      if (!exec(E)) { std::cerr << "ERR enc exec " << i << "\n"; return 1; }
      memcpy(&eo[(size_t)i * EO], fbuf(E.outputs[0]), EO * 4);
      memcpy(fbuf(D.inputs[0]), &din[(size_t)i * DI], DI * 4);
      if (!exec(D)) { std::cerr << "ERR dec exec " << i << "\n"; return 1; }
      memcpy(&dout[(size_t)i * DO], fbuf(D.outputs[0]), DO * 4);
    }
    writeAll(outDir + "/enc_raw.f32", eo.data(), eo.size());
    writeAll(outDir + "/dec_raw.f32", dout.data(), dout.size());
    std::cout << "CORRECT_DONE N=" << N << " perf_rc=" << perfRc << std::endl;
  } else {
    FILE *csv = fopen((outDir + "/lat.csv").c_str(), "w");
    fprintf(csv, "rep,i,enc_us,dec_us,glue_us,step_us,late_us,enc_acc_us,dec_acc_us\n");
    std::vector<float> acts((size_t)reps * DO);
    uint64_t next = nowUs();
    for (long r = -warm; r < reps; ++r) {
      long i = ((r % N) + N) % N;
      uint64_t late = 0;
      if (periodUs > 0) {
        uint64_t n0 = nowUs();
        if (n0 < next) std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::microseconds(next)));
        else late = n0 - next;
        next += periodUs;
      }
      uint64_t s0 = nowUs();
      memcpy(fbuf(E.inputs[0]), &ein[(size_t)i * EI], EI * 4);
      uint64_t a0 = nowUs(); bool ok = exec(E); uint64_t a1 = nowUs(); uint64_t eAcc = g_lastAccUs;
      if (!ok) { std::cerr << "ERR enc exec\n"; return 1; }
      float *dinb = fbuf(D.inputs[0]);
      memcpy(dinb, &din[(size_t)i * DI], DI * 4);
      const float *eo = fbuf(E.outputs[0]);
      if (encKind == "core") {
        int k = (int)ein[(size_t)i * EI + 0];
        if (k < 0 || k > 2) k = 0;
        for (int j = 0; j < 64; ++j) dinb[j] = std::nearbyint(eo[k * 64 + j]) / 16.0f;  // round-half-even == np.round
      } else {
        memcpy(dinb, eo, 64 * 4);
      }
      uint64_t b0 = nowUs(); ok = exec(D); uint64_t b1 = nowUs(); uint64_t dAcc = g_lastAccUs;
      if (!ok) { std::cerr << "ERR dec exec\n"; return 1; }
      if (r >= 0) memcpy(&acts[(size_t)r * DO], fbuf(D.outputs[0]), DO * 4);
      uint64_t s1 = nowUs();
      if (r >= 0)
        fprintf(csv, "%ld,%ld,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", r, i, (unsigned long)(a1 - a0), (unsigned long)(b1 - b0),
                (unsigned long)((s1 - s0) - (a1 - a0) - (b1 - b0)), (unsigned long)(s1 - s0), (unsigned long)late,
                (unsigned long)eAcc, (unsigned long)dAcc);
    }
    fclose(csv);
    writeAll(outDir + "/lat_actions.f32", acts.data(), acts.size());
    std::cout << "LATENCY_DONE reps=" << reps << " warm=" << warm << " period_us=" << periodUs << " perf_rc=" << perfRc << std::endl;
  }
  for (Loaded *L : {&E, &D}) {
    g_io.tearDownInputAndOutputTensors(L->inputs, L->outputs, (*L->graphsInfo)[0].numInputTensors, (*L->graphsInfo)[0].numOutputTensors);
    g_fp.qnnInterface.contextFree(L->context, nullptr);
  }
  if (g_device && g_fp.qnnInterface.deviceFree) g_fp.qnnInterface.deviceFree(g_device);
  if (g_backend && g_fp.qnnInterface.backendFree) g_fp.qnnInterface.backendFree(g_backend);
  return 0;
}
