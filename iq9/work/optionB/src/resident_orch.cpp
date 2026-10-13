//==============================================================================
// resident_orch.cpp — RESIDENT multi-context QNN graph executor for GR00T on the IQ-9075.
//
// Loads the 7 GR00T context binaries ONCE (shared backend + device), retrieves
// each graph, sets up input/output tensors ONCE, then serves commands on stdin so
// the context binaries stay DDR-resident across frames.
//
// Commands (line-based, stdin -> stdout):
//   LOADED <name>            (printed at startup, one per preloaded graph)
//   READY load_us=.. ngraphs=..
//   EXEC <graphname> <input_list_file> <output_dir>   ->  OK <graphname> <exec_us>
//   GFRAME <io_dir> <out_file> [pixel_path]           ->  OKGFRAME <k=v ...>
//   PING -> PONG ;  QUIT -> BYE (exit 0)
//   ERR <msg> on failure (server stays alive for EXEC/GFRAME errors).
//
// GFRAME runs GR00T's whole glued action loop IN-MEMORY: ViT -> [mlp1 GEMM +
// embed/scatter host glue] -> LLM -> vl-self-attn -> state-enc -> 4x{action-enc +
// pos-embed/concat glue -> DiT -> action-dec -> Euler}. All FLOAT_32 intermediates
// are memcpy'd graph->graph in the native client buffers (no .raw round-trips); the
// small INT_32 file inputs (position_ids, attention_mask) reuse the exact verified
// FLOAT->native cast. mlp1 is a single 256x1152 @ 1152x2048 Linear done via
// OpenBLAS cblas_sgemm (dlopen'd from torch's libopenblas at runtime, LP64).
//==============================================================================

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "DataUtil.hpp"
#include "DynamicLoadUtil.hpp"
#include "IOTensor.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleAppUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnWrapperUtils.hpp"
#include "SampleApp.hpp"

#include <dlfcn.h>

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
void *g_backendHandle = nullptr;
void *g_modelHandle = nullptr;
Qnn_LogHandle_t g_logHandle = nullptr;
Qnn_BackendHandle_t g_backend = nullptr;
Qnn_DeviceHandle_t g_device = nullptr;
iotensor::IOTensor g_io;
std::map<std::string, Loaded> g_graphs;

// ---- OpenBLAS (dlopen'd LP64 cblas_sgemm) ---------------------------------
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
typedef void (*sgemm_fn)(int, int, int, int, int, int, float, const float *, int,
                         const float *, int, float, float *, int);
typedef void (*setthreads_fn)(int);
sgemm_fn g_sgemm = nullptr;

// ---- GR00T glue constants (loaded once on first GFRAME) -------------------
struct Glue {
  bool loaded = false;
  std::vector<float> mlp1_w;    // (2048,1152) row-major
  std::vector<float> mlp1_b;    // (2048,)
  std::vector<float> future;    // (32,1536)
  std::vector<float> posemb;    // (1024,1536) -> use first 16 rows
  const float *embed = nullptr; // mmap (151680,2048) float32
  size_t embed_rows = 0;
  int image_token = 151669;
};
Glue g_glue;

// GR00T dims
const int NPATCH = 256, DVIT = 1152, DLLM = 2048;
const int DAH = 1536, HORIZON = 16, ADIM = 32, NFUT = 32, SA_ROWS = 49;
const int MO_D = 1024, PRED_ROWS = 49;
const int NSTEPS = 4;
const int EMBED_ROWS = 151680;

uint64_t nowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// read the float32 body of a .npy (128-byte header) into a vector
bool loadNpyF32(const std::string &path, std::vector<float> &out, size_t nElems) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) { std::cout << "ERR open npy " << path << std::endl; return false; }
  if (fseek(f, 128, SEEK_SET) != 0) { fclose(f); std::cout << "ERR seek " << path << std::endl; return false; }
  out.resize(nElems);
  size_t got = fread(out.data(), sizeof(float), nElems, f);
  fclose(f);
  if (got != nElems) { std::cout << "ERR short npy " << path << " " << got << "/" << nElems << std::endl; return false; }
  return true;
}

bool loadGlue(const std::string &glueDir) {
  if (g_glue.loaded) return true;
  if (!loadNpyF32(glueDir + "/mlp1_0_weight.npy", g_glue.mlp1_w, (size_t)DLLM * DVIT)) return false;
  if (!loadNpyF32(glueDir + "/mlp1_0_bias.npy", g_glue.mlp1_b, DLLM)) return false;
  if (!loadNpyF32(glueDir + "/ah_future_tokens_weight.npy", g_glue.future, (size_t)NFUT * DAH)) return false;
  if (!loadNpyF32(glueDir + "/ah_position_embedding_weight.npy", g_glue.posemb, (size_t)1024 * DAH)) return false;
  // mmap embed table (skip 128-byte npy header)
  std::string ep = glueDir + "/embed_tokens_weight.npy";
  int fd = open(ep.c_str(), O_RDONLY);
  if (fd < 0) { std::cout << "ERR open embed " << ep << std::endl; return false; }
  struct stat st; fstat(fd, &st);
  void *m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (m == MAP_FAILED) { std::cout << "ERR mmap embed" << std::endl; return false; }
  g_glue.embed = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(m) + 128);
  g_glue.embed_rows = EMBED_ROWS;
  if (const char *it = getenv("GROOT_IMAGE_TOKEN")) g_glue.image_token = atoi(it);
  g_glue.loaded = true;
  std::cerr << "[GLUE] loaded mlp1(" << g_glue.mlp1_w.size() << ") future(" << g_glue.future.size()
            << ") posemb(" << g_glue.posemb.size() << ") embed mmap rows=" << g_glue.embed_rows
            << " image_token=" << g_glue.image_token << "\n";
  return true;
}

// ---- tensor helpers --------------------------------------------------------
inline float *fbuf(Qnn_Tensor_t &t) { return static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(t).data); }
inline void *vbuf(Qnn_Tensor_t &t) { return QNN_TENSOR_GET_CLIENT_BUF(t).data; }
inline Qnn_DataType_t dtype(Qnn_Tensor_t &t) { return QNN_TENSOR_GET_DATA_TYPE(t); }
inline size_t telems(Qnn_Tensor_t &t) {
  uint32_t r = QNN_TENSOR_GET_RANK(t);
  uint32_t *d = QNN_TENSOR_GET_DIMENSIONS(t);
  size_t n = 1;
  for (uint32_t i = 0; i < r; ++i) n *= d[i];
  return n;
}
inline Qnn_Tensor_t *byName(Qnn_Tensor_t *arr, uint32_t n, const std::string &nm) {
  for (uint32_t i = 0; i < n; ++i)
    if (nm == std::string(QNN_TENSOR_GET_NAME(arr[i]))) return &arr[i];
  return nullptr;
}
inline uint32_t nIn(Loaded &L) { return (*L.graphsInfo)[0].numInputTensors; }
inline uint32_t nOut(Loaded &L) { return (*L.graphsInfo)[0].numOutputTensors; }

bool loadContext(const std::string &path) {
  tools::datautil::StatusCode ds;
  uint64_t bufSize = 0;
  std::tie(ds, bufSize) = tools::datautil::getFileSize(path);
  if (bufSize == 0) { std::cout << "ERR empty context file: " << path << std::endl; return false; }
  std::vector<uint8_t> buffer(bufSize);
  if (tools::datautil::readBinaryFromFile(path, buffer.data(), bufSize) !=
      tools::datautil::StatusCode::SUCCESS) {
    std::cout << "ERR read failed: " << path << std::endl; return false;
  }
  QnnSystemContext_Handle_t sysCtx = nullptr;
  if (g_fp.qnnSystemInterface.systemContextCreate(&sysCtx) != QNN_SUCCESS) {
    std::cout << "ERR systemContextCreate" << std::endl; return false;
  }
  const QnnSystemContext_BinaryInfo_t *binInfo = nullptr;
  Qnn_ContextBinarySize_t binInfoSize = 0;
  if (g_fp.qnnSystemInterface.systemContextGetBinaryInfo(
          sysCtx, buffer.data(), bufSize, &binInfo, &binInfoSize) != QNN_SUCCESS) {
    std::cout << "ERR systemContextGetBinaryInfo: " << path << std::endl; return false;
  }
  Loaded L;
  if (!sample_app::copyMetadataToGraphsInfo(binInfo, L.graphsInfo, L.graphsCount)) {
    std::cout << "ERR copyMetadataToGraphsInfo: " << path << std::endl; return false;
  }
  g_fp.qnnSystemInterface.systemContextFree(sysCtx);
  if (g_fp.qnnInterface.contextCreateFromBinary(
          g_backend, g_device, nullptr, buffer.data(), bufSize, &L.context, nullptr) != QNN_SUCCESS) {
    std::cout << "ERR contextCreateFromBinary: " << path << std::endl; return false;
  }
  for (uint32_t g = 0; g < L.graphsCount; ++g) {
    if (g_fp.qnnInterface.graphRetrieve(
            L.context, (*L.graphsInfo)[g].graphName, &((*L.graphsInfo)[g].graph)) != QNN_SUCCESS) {
      std::cout << "ERR graphRetrieve: " << path << std::endl; return false;
    }
  }
  if (g_io.setupInputAndOutputTensors(&L.inputs, &L.outputs, (*L.graphsInfo)[0]) !=
      iotensor::StatusCode::SUCCESS) {
    std::cout << "ERR setupInputAndOutputTensors: " << path << std::endl; return false;
  }
  L.name = (*L.graphsInfo)[0].graphName;
  g_graphs[L.name] = L;
  std::cout << "LOADED " << L.name << std::endl;
  return true;
}

void dumpGraphIO() {
  for (auto &kv : g_graphs) {
    Loaded &L = kv.second;
    std::cerr << "[DUMP] graph=" << L.name << " nIn=" << nIn(L) << " nOut=" << nOut(L) << "\n";
    for (uint32_t i = 0; i < nIn(L); ++i)
      std::cerr << "  IN  " << QNN_TENSOR_GET_NAME(L.inputs[i]) << " dtype="
                << (int)dtype(L.inputs[i]) << " elems=" << telems(L.inputs[i]) << "\n";
    for (uint32_t i = 0; i < nOut(L); ++i)
      std::cerr << "  OUT " << QNN_TENSOR_GET_NAME(L.outputs[i]) << " dtype="
                << (int)dtype(L.outputs[i]) << " elems=" << telems(L.outputs[i]) << "\n";
  }
  std::cerr.flush();
}

bool execOnly(Loaded &L, uint64_t &execUs) {
  uint64_t t0 = nowUs();
  Qnn_ErrorHandle_t e = g_fp.qnnInterface.graphExecute((*L.graphsInfo)[0].graph, L.inputs, nIn(L),
                                                       L.outputs, nOut(L), nullptr, nullptr);
  execUs = nowUs() - t0;
  return e == QNN_GRAPH_NO_ERROR;
}

// read n float32 from a .raw file
bool readF32(const std::string &path, std::vector<float> &out, size_t n) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) { std::cout << "ERR open " << path << std::endl; return false; }
  out.resize(n);
  size_t got = fread(out.data(), sizeof(float), n, f);
  fclose(f);
  if (got != n) { std::cout << "ERR short read " << path << " " << got << "/" << n << std::endl; return false; }
  return true;
}

// Fill a named input tensor from a float32 .raw file, honoring the tensor dtype.
// FLOAT_32 -> copy; INT_32 -> static_cast<int32_t> (byte-identical to the verified
// IOTensor FLOAT-populate path: copyFromFloatToNative / castFromFloat<int32_t>).
bool setInputFromF32File(Qnn_Tensor_t &t, const std::string &path) {
  size_t n = telems(t);
  std::vector<float> tmp;
  if (!readF32(path, tmp, n)) return false;
  Qnn_DataType_t dt = dtype(t);
  if (dt == QNN_DATATYPE_FLOAT_32) {
    memcpy(fbuf(t), tmp.data(), n * sizeof(float));
  } else if (dt == QNN_DATATYPE_INT_32) {
    int32_t *d = static_cast<int32_t *>(vbuf(t));
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(tmp[i]);
  } else {
    std::cout << "ERR unsupported input dtype " << (int)dt << std::endl; return false;
  }
  return true;
}

// set a single scalar input (timesteps_tensor) honoring dtype
void setScalar(Qnn_Tensor_t &t, float v) {
  if (dtype(t) == QNN_DATATYPE_INT_32) *static_cast<int32_t *>(vbuf(t)) = static_cast<int32_t>(v);
  else *fbuf(t) = v;
}

struct GTimes { uint64_t vit, mlp1, gluevlm, llm, vlln, state; uint64_t aenc, dit, adec; uint64_t glueah; };

bool runGrootFrame(const std::string &ioDir, const std::string &outFile,
                   const std::string &pixelPath, GTimes &tm) {
  Loaded &vit = g_graphs.at("groot_vit");
  Loaded &llm = g_graphs.at("groot_llm");
  Loaded &vlln = g_graphs.at("vlln_vl_self_attention_47");
  Loaded &state = g_graphs.at("state_encoder_47");
  Loaded &aenc = g_graphs.at("action_encoder_47");
  Loaded &dit = g_graphs.at("groot_dit");
  Loaded &adec = g_graphs.at("action_decoder_47");
  uint64_t g0;

  // read input_ids (int64) -> N tokens
  std::string idp = ioDir + "/input_ids.raw";
  FILE *fi = fopen(idp.c_str(), "rb");
  if (!fi) { std::cout << "ERR open input_ids" << std::endl; return false; }
  fseek(fi, 0, SEEK_END); long ibytes = ftell(fi); fseek(fi, 0, SEEK_SET);
  size_t N = ibytes / 8;
  std::vector<int64_t> ids(N);
  if (fread(ids.data(), 8, N, fi) != N) { fclose(fi); std::cout << "ERR read input_ids" << std::endl; return false; }
  fclose(fi);

  // --- ViT ---
  { Qnn_Tensor_t *pv = byName(vit.inputs, nIn(vit), "pixel_values");
    Qnn_Tensor_t *pid = byName(vit.inputs, nIn(vit), "position_ids");
    if (!pv || !pid) { std::cout << "ERR vit inputs" << std::endl; return false; }
    if (!setInputFromF32File(*pv, pixelPath)) return false;
    if (!setInputFromF32File(*pid, ioDir + "/position_ids.raw")) return false; }
  if (!execOnly(vit, tm.vit)) { std::cout << "ERR exec vit" << std::endl; return false; }
  float *vit_out = fbuf(vit.outputs[0]);  // (256,1152)
  if (telems(vit.outputs[0]) != (size_t)NPATCH * DVIT) { std::cout << "ERR vit out size" << std::endl; return false; }

  // --- mlp1 GEMM: proj[256,2048] = vit_out[256,1152] @ W[2048,1152]^T + b ---
  std::vector<float> proj((size_t)NPATCH * DLLM);
  g0 = nowUs();
  g_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NPATCH, DLLM, DVIT, 1.0f,
          vit_out, DVIT, g_glue.mlp1_w.data(), DVIT, 0.0f, proj.data(), DLLM);
  for (int i = 0; i < NPATCH; ++i) {
    float *pr = &proj[(size_t)i * DLLM];
    for (int j = 0; j < DLLM; ++j) pr[j] += g_glue.mlp1_b[j];
  }
  tm.mlp1 = nowUs() - g0;

  // --- embed lookup + image-token scatter -> inputs_embeds[N,2048], into LLM buf ---
  Qnn_Tensor_t *ie = byName(llm.inputs, nIn(llm), "inputs_embeds");
  Qnn_Tensor_t *am = byName(llm.inputs, nIn(llm), "attention_mask");
  if (!ie || !am) { std::cout << "ERR llm inputs" << std::endl; return false; }
  if (telems(*ie) != N * DLLM) { std::cout << "ERR inputs_embeds size " << telems(*ie) << "/" << N * DLLM << std::endl; return false; }
  g0 = nowUs();
  { float *dst = fbuf(*ie);
    size_t pk = 0;
    for (size_t i = 0; i < N; ++i) {
      int64_t id = ids[i];
      const float *src;
      if (id == g_glue.image_token) { src = &proj[pk * DLLM]; ++pk; }
      else { src = &g_glue.embed[(size_t)id * DLLM]; }
      memcpy(&dst[i * DLLM], src, DLLM * sizeof(float));
    } }
  tm.gluevlm = nowUs() - g0;
  if (!setInputFromF32File(*am, ioDir + "/attention_mask.raw")) return false;

  // --- LLM ---
  if (!execOnly(llm, tm.llm)) { std::cout << "ERR exec llm" << std::endl; return false; }

  // --- vl-self-attn: backbone -> vl_embs ---
  { Qnn_Tensor_t *bf = byName(vlln.inputs, nIn(vlln), "backbone_features");
    if (!bf || telems(*bf) != telems(llm.outputs[0])) { std::cout << "ERR vlln in" << std::endl; return false; }
    memcpy(fbuf(*bf), fbuf(llm.outputs[0]), telems(*bf) * sizeof(float)); }
  if (!execOnly(vlln, tm.vlln)) { std::cout << "ERR exec vlln" << std::endl; return false; }

  // copy vl_embs -> DiT vl_embs input ONCE (constant across denoise steps)
  { Qnn_Tensor_t *dvl = byName(dit.inputs, nIn(dit), "vl_embs");
    if (!dvl || telems(*dvl) != telems(vlln.outputs[0])) {
      std::cout << "ERR dit vl_embs size " << (dvl ? telems(*dvl) : 0) << "/" << telems(vlln.outputs[0]) << std::endl; return false; }
    memcpy(fbuf(*dvl), fbuf(vlln.outputs[0]), telems(*dvl) * sizeof(float)); }

  // --- state_encoder ---
  { Qnn_Tensor_t *s = byName(state.inputs, nIn(state), "state");
    if (!s) { std::cout << "ERR state in" << std::endl; return false; }
    if (!setInputFromF32File(*s, ioDir + "/state.raw")) return false; }
  if (!execOnly(state, tm.state)) { std::cout << "ERR exec state" << std::endl; return false; }
  float *state_feat = fbuf(state.outputs[0]);  // (1,1,1536)
  if (telems(state.outputs[0]) != (size_t)DAH) { std::cout << "ERR state out size" << std::endl; return false; }

  // --- denoise loop x4 ---
  std::vector<float> actions;
  if (!readF32(ioDir + "/init_actions.raw", actions, (size_t)HORIZON * ADIM)) return false;
  Qnn_Tensor_t *aA = byName(aenc.inputs, nIn(aenc), "actions");
  Qnn_Tensor_t *aTP = byName(aenc.inputs, nIn(aenc), "time_proj");
  Qnn_Tensor_t *dSA = byName(dit.inputs, nIn(dit), "sa_embs");
  Qnn_Tensor_t *dTS = byName(dit.inputs, nIn(dit), "timesteps_tensor");
  Qnn_Tensor_t *aMO = byName(adec.inputs, nIn(adec), "model_output");
  if (!aA || !aTP || !dSA || !dTS || !aMO) { std::cout << "ERR loop io" << std::endl; return false; }
  tm.aenc = tm.dit = tm.adec = tm.glueah = 0;
  const float dt = 1.0f / (float)NSTEPS;  // POSITIVE 0.25
  std::vector<float> sa((size_t)SA_ROWS * DAH);

  for (int s = 0; s < NSTEPS; ++s) {
    int td = (int)(((double)s / (double)NSTEPS) * 1000.0);  // 0,250,500,750
    memcpy(fbuf(*aA), actions.data(), (size_t)HORIZON * ADIM * sizeof(float));
    char tp[512]; snprintf(tp, sizeof(tp), "%s/time_proj_t%d.raw", ioDir.c_str(), td);
    if (!setInputFromF32File(*aTP, tp)) return false;
    uint64_t e;
    if (!execOnly(aenc, e)) { std::cout << "ERR exec aenc" << std::endl; return false; } tm.aenc += e;
    float *af = fbuf(aenc.outputs[0]);  // (1,16,1536)

    g0 = nowUs();
    // sa_embs = concat(state(1,1536), future(32,1536), af+posemb[:16](16,1536))
    memcpy(&sa[0], state_feat, (size_t)DAH * sizeof(float));
    memcpy(&sa[(size_t)DAH], g_glue.future.data(), (size_t)NFUT * DAH * sizeof(float));
    for (int r = 0; r < HORIZON; ++r) {
      float *dstr = &sa[(size_t)(1 + NFUT + r) * DAH];
      float *afr = &af[(size_t)r * DAH];
      const float *pe = &g_glue.posemb[(size_t)r * DAH];
      for (int j = 0; j < DAH; ++j) dstr[j] = afr[j] + pe[j];
    }
    memcpy(fbuf(*dSA), sa.data(), sa.size() * sizeof(float));
    setScalar(*dTS, (float)td);
    tm.glueah += nowUs() - g0;

    if (!execOnly(dit, e)) { std::cout << "ERR exec dit" << std::endl; return false; } tm.dit += e;
    { float *mo = fbuf(dit.outputs[0]);
      if (telems(dit.outputs[0]) != (size_t)PRED_ROWS * MO_D) { std::cout << "ERR dit out size" << std::endl; return false; }
      memcpy(fbuf(*aMO), mo, (size_t)PRED_ROWS * MO_D * sizeof(float)); }
    if (!execOnly(adec, e)) { std::cout << "ERR exec adec" << std::endl; return false; } tm.adec += e;
    float *pred = fbuf(adec.outputs[0]);  // (1,49,32)

    g0 = nowUs();
    // euler: actions += dt * pred[:, -16:]  (last 16 of 49 rows)
    const int off = PRED_ROWS - HORIZON;  // 33
    for (int r = 0; r < HORIZON; ++r) {
      float *ar = &actions[(size_t)r * ADIM];
      float *vr = &pred[(size_t)(off + r) * ADIM];
      for (int j = 0; j < ADIM; ++j) ar[j] += dt * vr[j];
    }
    tm.glueah += nowUs() - g0;
  }

  FILE *fo = fopen(outFile.c_str(), "wb");
  if (!fo) { std::cout << "ERR open out " << outFile << std::endl; return false; }
  fwrite(actions.data(), sizeof(float), actions.size(), fo);
  fclose(fo);
  return true;
}

// ---- generic EXEC (file-based, kept for debug/verification) ---------------
bool execGraph(const std::string &name, const std::string &listFile, const std::string &outDir,
               uint64_t &execUs) {
  auto it = g_graphs.find(name);
  if (it == g_graphs.end()) { std::cout << "ERR unknown graph: " << name << std::endl; return false; }
  Loaded &L = it->second;
  std::vector<std::vector<std::vector<std::string>>> fileLists;
  std::vector<std::unordered_map<std::string, uint32_t>> nameToIdx;
  bool ok = false;
  std::tie(fileLists, nameToIdx, ok) = sample_app::readInputLists({listFile});
  if (!ok || fileLists.empty() || fileLists[0].empty()) { std::cout << "ERR readInputLists: " << listFile << std::endl; return false; }
  iotensor::StatusCode s; size_t numPop = 0, batch = 0;
  std::tie(s, numPop, batch) = g_io.populateInputTensors(
      0, fileLists[0], 0, false, nameToIdx[0], L.inputs, (*L.graphsInfo)[0], iotensor::InputDataType::FLOAT);
  if (s != iotensor::StatusCode::SUCCESS) { std::cout << "ERR populateInputTensors: " << name << std::endl; return false; }
  uint64_t t0 = nowUs();
  Qnn_ErrorHandle_t e = g_fp.qnnInterface.graphExecute((*L.graphsInfo)[0].graph, L.inputs, nIn(L),
                                                       L.outputs, nOut(L), nullptr, nullptr);
  execUs = nowUs() - t0;
  if (e != QNN_GRAPH_NO_ERROR) { std::cout << "ERR graphExecute rc=" << e << " graph=" << name << std::endl; return false; }
  if (g_io.writeOutputTensors(0, 0, (*L.graphsInfo)[0].graphName, L.outputs, nOut(L),
                              iotensor::OutputDataType::FLOAT_ONLY, 1, outDir, numPop, batch) !=
      iotensor::StatusCode::SUCCESS) {
    std::cout << "ERR writeOutputTensors: " << name << std::endl; return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  std::string backendPath, systemLib, blasPath, glueDir = "/opt/gvrb/optionB/glue_constants";  // = $IQ9_WORK default; override with --glue
  std::vector<std::string> ctxPaths;
  QnnLog_Level_t logLevel = QNN_LOG_LEVEL_ERROR;
  int blasThreads = 6;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--backend" && i + 1 < argc) backendPath = argv[++i];
    else if (a == "--system_library" && i + 1 < argc) systemLib = argv[++i];
    else if (a == "--ctx" && i + 1 < argc) ctxPaths.push_back(argv[++i]);
    else if (a == "--blas" && i + 1 < argc) blasPath = argv[++i];
    else if (a == "--glue" && i + 1 < argc) glueDir = argv[++i];
    else if (a == "--blas_threads" && i + 1 < argc) blasThreads = atoi(argv[++i]);
    else if (a == "--log_level" && i + 1 < argc) {
      std::string l = argv[++i];
      if (l == "warn") logLevel = QNN_LOG_LEVEL_WARN;
      else if (l == "info") logLevel = QNN_LOG_LEVEL_INFO;
      else if (l == "verbose") logLevel = QNN_LOG_LEVEL_VERBOSE;
    }
  }
  if (backendPath.empty() || systemLib.empty() || ctxPaths.empty()) {
    std::cerr << "usage: resident_orch --backend <lib> --system_library <lib> "
                 "--ctx name.bin [--ctx ...] [--blas libopenblas.so] [--glue dir]\n";
    return 2;
  }

  // dlopen OpenBLAS (LP64 cblas_sgemm) for the mlp1 projection
  if (!blasPath.empty()) {
    void *bh = dlopen(blasPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!bh) { std::cerr << "ERR dlopen blas: " << dlerror() << "\n"; return 1; }
    g_sgemm = (sgemm_fn)dlsym(bh, "cblas_sgemm");
    if (!g_sgemm) { std::cerr << "ERR dlsym cblas_sgemm\n"; return 1; }
    setthreads_fn st = (setthreads_fn)dlsym(bh, "openblas_set_num_threads");
    if (st) st(blasThreads);
    std::cerr << "[BLAS] cblas_sgemm loaded, threads=" << blasThreads << "\n";
  }

  if (!qnn::log::initializeLogging()) { std::cerr << "ERR init logging\n"; return 1; }
  qnn::log::setLogLevel(logLevel);

  if (dynamicloadutil::getQnnFunctionPointers(backendPath, "", &g_fp, &g_backendHandle, false, &g_modelHandle) !=
      dynamicloadutil::StatusCode::SUCCESS) { std::cerr << "ERR load backend fn pointers\n"; return 1; }
  if (dynamicloadutil::getQnnSystemFunctionPointers(systemLib, &g_fp) != dynamicloadutil::StatusCode::SUCCESS) {
    std::cerr << "ERR load system fn pointers\n"; return 1;
  }
  if (g_fp.qnnInterface.logCreate)
    g_fp.qnnInterface.logCreate(qnn::log::getLogCallback(), qnn::log::getLogLevel(), &g_logHandle);
  if (g_fp.qnnInterface.backendCreate(g_logHandle, nullptr, &g_backend) != QNN_BACKEND_NO_ERROR) {
    std::cerr << "ERR backendCreate\n"; return 1;
  }
  if (g_fp.qnnInterface.deviceCreate) {
    Qnn_ErrorHandle_t de = g_fp.qnnInterface.deviceCreate(g_logHandle, nullptr, &g_device);
    if (de != QNN_SUCCESS && de != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) { std::cerr << "ERR deviceCreate rc=" << de << "\n"; return 1; }
  }

  uint64_t tLoad0 = nowUs();
  for (auto &p : ctxPaths) {
    if (!loadContext(p)) { std::cerr << "ERR failed to load context: " << p << "\n"; return 1; }
  }
  uint64_t loadUs = nowUs() - tLoad0;
  dumpGraphIO();
  if (g_sgemm) { if (!loadGlue(glueDir)) { std::cerr << "ERR loadGlue\n"; return 1; } }
  std::cout << "READY load_us=" << loadUs << " ngraphs=" << g_graphs.size() << std::endl;
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::vector<std::string> tok;
    sample_app::split(tok, line, ' ');
    if (tok.empty()) continue;
    if (tok[0] == "QUIT") { std::cout << "BYE" << std::endl; break; }
    else if (tok[0] == "PING") { std::cout << "PONG" << std::endl; }
    else if (tok[0] == "EXEC" && tok.size() >= 4) {
      uint64_t execUs = 0;
      if (execGraph(tok[1], tok[2], tok[3], execUs)) std::cout << "OK " << tok[1] << " " << execUs << std::endl;
    } else if (tok[0] == "GFRAME" && tok.size() >= 3) {
      std::string pixelPath = (tok.size() >= 4) ? tok[3] : (tok[1] + "/pixel_values.raw");
      GTimes tm; memset(&tm, 0, sizeof(tm));
      uint64_t w0 = nowUs();
      if (runGrootFrame(tok[1], tok[2], pixelPath, tm)) {
        uint64_t wall = nowUs() - w0;
        std::cout << "OKGFRAME wall_us=" << wall << " vit=" << tm.vit << " mlp1=" << tm.mlp1
                  << " gluevlm=" << tm.gluevlm << " llm=" << tm.llm << " vlln=" << tm.vlln
                  << " state=" << tm.state << " aenc_sum=" << tm.aenc << " dit_sum=" << tm.dit
                  << " adec_sum=" << tm.adec << " glueah=" << tm.glueah << std::endl;
      }
    } else { std::cout << "ERR bad command: " << line << std::endl; }
    std::cout.flush();
  }

  for (auto &kv : g_graphs) {
    Loaded &L = kv.second;
    g_io.tearDownInputAndOutputTensors(L.inputs, L.outputs, (*L.graphsInfo)[0].numInputTensors,
                                       (*L.graphsInfo)[0].numOutputTensors);
    if (L.context) g_fp.qnnInterface.contextFree(L.context, nullptr);
  }
  if (g_device && g_fp.qnnInterface.deviceFree) g_fp.qnnInterface.deviceFree(g_device);
  if (g_backend && g_fp.qnnInterface.backendFree) g_fp.qnnInterface.backendFree(g_backend);
  if (g_logHandle && g_fp.qnnInterface.logFree) g_fp.qnnInterface.logFree(g_logHandle);
  if (g_backendHandle) pal::dynamicloading::dlClose(g_backendHandle);
  return 0;
}
