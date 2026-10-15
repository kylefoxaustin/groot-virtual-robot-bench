// rtctl.hpp -- for rt/sonic_rt.cpp (groot-virtual-robot-bench; not upstream): per-thread scheduling knobs read from the environment,
// applied by the thread ITSELF, plus thread naming so a scheduler trace / /proc readback can tell threads apart.
//   <P>_CPUS  e.g. "1" or "4-6"  -> sched_setaffinity(calling thread)
//   <P>_FIFO  e.g. "90"          -> SCHED_FIFO at that priority (calling thread)
//   <P>_NICE  e.g. "10"          -> setpriority(PRIO_PROCESS, tid) (calling thread, SCHED_OTHER only)
// With none of the variables set, apply() only names the thread (no scheduling change).
// Every application is logged to stderr with its return codes ("[rtctl] ...") so the run log carries the proof.
#pragma once
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <onnxruntime_c_api.h>

namespace rtctl {
inline pid_t tid() { return (pid_t)syscall(SYS_gettid); }
inline bool parse_cpus(const char* s, cpu_set_t& cs) {
  CPU_ZERO(&cs); bool any = false;
  std::string str(s); size_t p = 0;
  while (p < str.size()) {
    size_t q = str.find(',', p); if (q == std::string::npos) q = str.size();
    std::string tok = str.substr(p, q - p); size_t d = tok.find('-');
    int a = std::atoi(tok.c_str()), b = (d == std::string::npos) ? a : std::atoi(tok.c_str() + d + 1);
    for (int c = a; c <= b; ++c) { CPU_SET(c, &cs); any = true; }
    p = q + 1;
  }
  return any;
}
inline void apply(const char* prefix, const char* name) {
  if (tid() != getpid()) pthread_setname_np(pthread_self(), name);  // never rename the main thread (= process comm)
  char k[96]; std::string log;
  std::snprintf(k, sizeof(k), "%s_CPUS", prefix);
  if (const char* v = std::getenv(k)) if (*v) {
    cpu_set_t cs; int rc = -99;
    if (parse_cpus(v, cs)) rc = sched_setaffinity(0, sizeof(cs), &cs);
    log += std::string(" cpus=") + v + " rc=" + std::to_string(rc) + (rc ? std::string("/") + std::strerror(errno) : "");
  }
  std::snprintf(k, sizeof(k), "%s_FIFO", prefix);
  if (const char* v = std::getenv(k)) if (*v) {
    sched_param sp{}; sp.sched_priority = std::atoi(v);
    int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    log += std::string(" fifo=") + v + " rc=" + std::to_string(rc);
  }
  std::snprintf(k, sizeof(k), "%s_NICE", prefix);
  if (const char* v = std::getenv(k)) if (*v) {
    int rc = setpriority(PRIO_PROCESS, tid(), std::atoi(v));
    log += std::string(" nice=") + v + " rc=" + std::to_string(rc);
  }
  std::fprintf(stderr, "[rtctl] %s tid=%d name=%s%s\n", prefix, (int)tid(), name, log.empty() ? " (name only)" : log.c_str());
}
// onnxruntime custom thread creation: every intra-op pool thread names itself and applies SONIC_ORT_* knobs.
struct OrtThr { std::thread t; };
inline OrtCustomThreadHandle ort_create(void*, OrtThreadWorkerFn fn, void* param) {
  auto* h = new OrtThr();
  h->t = std::thread([fn, param]() { apply("SONIC_ORT", "sonic_ort"); fn(param); });
  return reinterpret_cast<OrtCustomThreadHandle>(h);
}
inline void ort_join(OrtCustomThreadHandle h) {
  auto* o = reinterpret_cast<OrtThr*>(const_cast<OrtCustomHandleType*>(h)); if (o->t.joinable()) o->t.join(); delete o;
}
}  // namespace rtctl
