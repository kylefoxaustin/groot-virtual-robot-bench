// sonic_probe.hpp -- per-tick measurement probe for the groot-virtual-robot-bench SONIC benchmark (NOT upstream).
// Records, for every CONTROL tick: steady-clock timestamps, encoder/decoder/whole-step durations,
// reference vs measured joint positions (IsaacLab order), and the exact encoder/decoder input and
// output tensors (float32). Everything is appended to preallocated memory inside the control loop
// (memcpy only, no I/O) and written to $SONIC_PROBE_DIR at process exit.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct SonicTickRec {
  uint64_t tick;
  int64_t t_start_ns;   // steady_clock at obs_start of the tick
  int64_t enc_ns;       // encoder_engine_->Encode() wall (incl. H2D, TRT, D2H, sync)
  int64_t dec_ns;       // policy_engine_->Infer() wall (incl. H2D, TRT, D2H, sync)
  int64_t step_ns;      // obs_start -> after CurrentFrameAdvancement (whole control step)
  int32_t enc_mode;
  int32_t frame;
  int32_t planner_active;
  int32_t play;
  float ref_q[29];      // reference joint pos (IsaacLab order) at current frame
  float meas_q[29];     // measured joint pos (IsaacLab order) from LowState used this tick
  float ref_root[3];    // reference root (body 0) position at current frame
  float pad_;
};

// step-2 addition: exact planner I/O per call (planner thread; memcpy only, after inference returns)
struct SonicPlannerIO {
  int64_t t_start_ns; int32_t kind; int32_t mode; int32_t seed; int32_t npf;
  float target_vel; float height; float mv[3]; float face[3];
  float ctx[4 * 36]; float qpos[64 * 36];
};
struct SonicPlannerRec { int64_t t_start_ns; int64_t total_ns; int64_t model_ns; int32_t kind; int32_t pad; };

class SonicProbe {
 public:
  static constexpr size_t ENC_IN = 1762, ENC_OUT = 64, DEC_IN = 994, DEC_OUT = 29;
  static constexpr size_t MAX_TICKS = 30000;  // 600 s at 50 Hz
  SonicProbe() {
    const char* d = std::getenv("SONIC_PROBE_DIR");
    enabled_ = d != nullptr;
    if (!enabled_) return;
    dir_ = d;
    recs_.reserve(MAX_TICKS); planner_.reserve(20000);
    enc_in_.resize(MAX_TICKS * ENC_IN); enc_out_.resize(MAX_TICKS * ENC_OUT);
    dec_in_.resize(MAX_TICKS * DEC_IN); dec_out_.resize(MAX_TICKS * DEC_OUT);
  }
  ~SonicProbe() { Dump(); }
  bool enabled() const { return enabled_; }
  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  // current-tick scratch (control thread only)
  SonicTickRec cur{};
  void SetEnc(const float* in, const float* out, int64_t ns) {
    if (!enabled_ || recs_.size() >= MAX_TICKS) return;
    size_t i = recs_.size();
    std::memcpy(&enc_in_[i * ENC_IN], in, ENC_IN * 4); std::memcpy(&enc_out_[i * ENC_OUT], out, ENC_OUT * 4);
    cur.enc_ns = ns;
  }
  void SetDec(const float* in, const float* out, int64_t ns) {
    if (!enabled_ || recs_.size() >= MAX_TICKS) return;
    size_t i = recs_.size();
    std::memcpy(&dec_in_[i * DEC_IN], in, DEC_IN * 4); std::memcpy(&dec_out_[i * DEC_OUT], out, DEC_OUT * 4);
    cur.dec_ns = ns;
  }
  void Commit() {
    if (!enabled_ || recs_.size() >= MAX_TICKS) return;
    cur.tick = recs_.size(); recs_.push_back(cur); cur = SonicTickRec{};
  }
  void Planner(int64_t t0, int64_t total, int64_t model, int kind) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> l(pm_);
    if (planner_.size() < 20000) planner_.push_back({t0, total, model, kind, 0});
  }
  template <class P> void PlannerIO(int64_t t0, int kind, P* p) {
    if (!enabled_ || !p) return;
    std::lock_guard<std::mutex> l(pm_);
    if (pio_.size() >= 4000) return;
    SonicPlannerIO r{}; r.t_start_ns = t0; r.kind = kind; r.mode = p->GetModeValue(); r.seed = p->GetRandomSeedValue();
    r.npf = p->GetNumPredFrames(); r.target_vel = p->GetTargetVelValue(); r.height = p->GetHeightValue();
    std::memcpy(r.mv, p->GetMovementDirectionValues(), 12); std::memcpy(r.face, p->GetFacingDirectionValues(), 12);
    std::memcpy(r.ctx, p->GetContextBuffer(), sizeof(r.ctx)); std::memcpy(r.qpos, p->GetMujocoQposBuffer(), sizeof(r.qpos));
    pio_.push_back(r);
  }
  void Dump() {
    if (!enabled_ || dumped_.exchange(true)) return;
    size_t n = recs_.size();
    auto w = [&](const std::string& name, const void* p, size_t bytes) {
      FILE* f = std::fopen((dir_ + "/" + name).c_str(), "wb");
      if (!f) return; std::fwrite(p, 1, bytes, f); std::fclose(f);
    };
    w("ticks.bin", recs_.data(), n * sizeof(SonicTickRec));
    w("enc_in.f32", enc_in_.data(), n * ENC_IN * 4); w("enc_out.f32", enc_out_.data(), n * ENC_OUT * 4);
    w("dec_in.f32", dec_in_.data(), n * DEC_IN * 4); w("dec_out.f32", dec_out_.data(), n * DEC_OUT * 4);
    { std::lock_guard<std::mutex> l(pm_); w("planner.bin", planner_.data(), planner_.size() * sizeof(SonicPlannerRec));
      w("planner_io.bin", pio_.data(), pio_.size() * sizeof(SonicPlannerIO)); }
    FILE* f = std::fopen((dir_ + "/probe_meta.txt").c_str(), "w");
    if (f) { std::fprintf(f, "ticks %zu\nplanner_calls %zu\nsizeof_tick %zu\nsizeof_planner %zu\nplanner_io %zu\nsizeof_planner_io %zu\n", n, planner_.size(), sizeof(SonicTickRec), sizeof(SonicPlannerRec), pio_.size(), sizeof(SonicPlannerIO)); std::fclose(f); }
    std::fprintf(stderr, "[SonicProbe] dumped %zu ticks, %zu planner calls to %s\n", n, planner_.size(), dir_.c_str());
  }
 private:
  bool enabled_ = false; std::atomic<bool> dumped_{false}; std::string dir_;
  std::vector<SonicTickRec> recs_; std::vector<SonicPlannerRec> planner_; std::vector<SonicPlannerIO> pio_; std::mutex pm_;
  std::vector<float> enc_in_, enc_out_, dec_in_, dec_out_;
};
inline SonicProbe g_sonic_probe;
