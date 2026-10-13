/* HMX matrix-engine kernels — isolated TU so only THIS file gets -mhmx (-mhmx on the HVX-heavy
   calculator_imp.c hangs the compiler backend). v73.

   DEFENSIVE: writes a STAGE marker to shared DDR c[25] before every risky step, so if an HMX
   instruction faults (which can wedge the PD), the host reads c[25] and knows exactly which op died.
   Uses Q6_mxmem_AR_after_hf for the store (single unambiguous hf def; Q6_mxmem_cvt_RR resolves to the
   fp8 builtin in this header). */
#include <string.h>
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_perf.h"

#define ST(n) do{ c[25]=(n); }while(0)   /* stage marker in uncached shared DDR */

static unsigned short f2h(float f){ __fp16 h=(__fp16)f; unsigned short u; memcpy(&u,&h,2); return u; }
static __attribute__((unused)) float h2f(unsigned short u){ __fp16 h; memcpy(&h,&u,2); return (float)h; }

/* Known-answer test: A[sp,ch]=sp+ch*0.01, W=identity*out -> Y[sp,out]=(sp+out*0.01)*out.
   One 32x32x32 fp16 HMX tile in VTCM; Y (float) -> shared DDR at byte 4096. Returns status (<=0 bad). */
static int g_pwr_anchor;
int hmx_selftest(volatile unsigned int* c){
  ST(1);
  /* REQUIRED: power up HMX. Without it HMX is enabled (ops execute) but the datapath isn't
     clocked -> the :after store is posted but never retires -> scalar read of the result line hangs. */
  HAP_power_request_t preq; memset(&preq,0,sizeof(preq));
  preq.type = HAP_power_set_HMX; preq.hmx.power_up = 1;
  int prc = HAP_power_set((void*)&g_pwr_anchor, &preq);
  c[29] = 0x50000000u | (unsigned)(prc & 0xFFFF);   /* record power-vote rc (low16) */
  compute_res_attr_t attr;
  if(compute_resource_attr_init(&attr)!=0) return -1;
  ST(2);
  if(compute_resource_attr_set_vtcm_param(&attr, 64*1024, 0)!=0) return -2;  /* 64KB, multi-page (single-page capped at ~4KB -> result@6144 was unmapped -> read hang) */
  ST(3);
  unsigned int ctx = HAP_compute_res_acquire(&attr, 500000);
  if(!ctx) return -3;
  ST(4);
  void* vtcm = compute_resource_attr_get_vtcm_ptr(&attr);
  if(!vtcm){ HAP_compute_res_release(ctx); return -4; }
  ST(5);
  unsigned char* base=(unsigned char*)vtcm;
  unsigned short* act=(unsigned short*)(base+0);
  unsigned short* wgt=(unsigned short*)(base+2048);
  unsigned char*  res=base+3*2048;
  int vec,ch,s,out,iv;
  for(vec=0;vec<16;vec++) for(ch=0;ch<32;ch++) for(s=0;s<2;s++){
    int sp=vec*2+s; act[vec*64+ch*2+s]=f2h((float)sp + (float)ch*0.01f); }
  for(vec=0;vec<16;vec++) for(out=0;out<32;out++) for(iv=0;iv<2;iv++){
    int in=vec*2+iv; wgt[vec*64+out*2+iv]=f2h((in==out)?(float)out:0.f); }
  memset(res,0,2048);
  int* bias=(int*)(base+4096); memset(bias,0,256);          /* column table: try scale=1.0 (fp16 0x3c00) in low16, bias=0 in high16 */
  for(out=0;out<32;out++) bias[out]=0x00003C00;
  ST(6);
  compute_res_hmx_mutex_t mtx;
  if(HAP_compute_res_hmx_lock3(ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &mtx, 500000)!=0){ HAP_compute_res_release(ctx); return -5; }
  ST(7);
  /* EXACT llama.cpp htp HMX sequence (ggml-hexagon/htp/hmx-utils.h + hmx-mm-kernels-tiled.h):
     bias set once; mxclracc; activation+weight co-issued in ONE VLIW packet {}; store :after.hf=acc.
     range=2047 (single 2KB tile) for BOTH act and weight; scale reg index 0. Issuing act and weight in
     separate packets leaves HMX malformed -> the store never completes -> the res read hangs. */
  unsigned int uact=(unsigned int)(unsigned long)act, uwgt=(unsigned int)(unsigned long)wgt;
  unsigned int ures=(unsigned int)(unsigned long)res, ubias=(unsigned int)(unsigned long)bias;
  ST(10);
  asm volatile("bias = mxmem(%0)\n" :: "r"(ubias) : "memory");           /* mxmem (NOT mxmem2)  */
  asm volatile("mxclracc.hf\n");
  ST(12);
  asm volatile("{\n activation.hf = mxmem(%1, %0):deep\n weight.hf = mxmem(%2, %0)\n}\n"
               :: "r"(2047u), "r"(uact), "r"(uwgt));                      /* :deep on activation, range=2047 for 1 tile */
  ST(13);
  asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(ures), "r"(0u) : "memory");
  ST(14); HAP_compute_res_hmx_unlock3(ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &mtx);
  ST(15);
  /* NON-WEDGING PROBE: first confirm a scalar read of VTCM WE wrote (act) works -- act[1]=fp16(1.0)=0x3c00.
     Then one probe read of the HMX-written result res[0] (if THIS hangs, it's HMX-region specific). */
  unsigned short* a16=(unsigned short*)act;
  c[27]=0xAC000000u | a16[1];          /* scalar VTCM read sanity (expect low16=0x3c00) */
  ST(16);
  unsigned short* r16=(unsigned short*)res;
  float* yout=(float*)((unsigned char*)c + 4096);
  int sp,o;
  for(sp=0;sp<32;sp++){ c[26]=sp+1; for(o=0;o<32;o++) yout[sp*32+o]=h2f(r16[(sp/2)*64 + o*2 + (sp%2)]); }
  ST(100);
  HAP_compute_res_release(ctx);
  return 0;
}

#define HMX_IDX(i,j) (64*((i)/2) + 2*(j) + ((i)%2))
/* cross-NSP barrier via shared uncached DDR (c[40]=NSP0 phase, c[41]=NSP1 phase) */
static inline void hmx_barrier(volatile unsigned int* c, unsigned int role, unsigned int* ph){
  (*ph)++; c[40+role]=*ph; unsigned s=0; while(c[40+(1u-role)] < *ph){ if(++s>2000000000u) return; }
}
/* HMX tiled fp16 GEMM: C[M,N] = A[M,K] @ W[K,N]. A/W/C fp16 in shared DDR (A@4096, W after A, C after W).
   Packs row-major -> VTCM crouton tiles, streams all K-tiles per output tile with :deep (one packet).
   Writes HMX-compute us to c[30], pack us to c[31]. M,K,N from c[14],c[15],c[16] (each mult of 32). */
int hmx_gemm(volatile unsigned int* c){
  int M=(int)c[14], K=(int)c[15], N=(int)c[16];
  int MT=M/32, KT=K/32, NT=N/32, mt,kt,nt,r,k,o;
  HAP_power_request_t preq; memset(&preq,0,sizeof(preq));
  preq.type=HAP_power_set_HMX; preq.hmx.power_up=1; HAP_power_set((void*)&g_pwr_anchor,&preq);
  compute_res_attr_t attr;
  if(compute_resource_attr_init(&attr)) return -1;
  unsigned int vneed=(unsigned)(((size_t)MT*KT + (size_t)KT*NT + (size_t)MT*NT)*2048 + 2048);
  if(compute_resource_attr_set_vtcm_param(&attr, vneed, 0)) return -2;
  unsigned int ctx=HAP_compute_res_acquire(&attr,1000000); if(!ctx) return -3;
  unsigned char* vb=(unsigned char*)compute_resource_attr_get_vtcm_ptr(&attr);
  if(!vb){ HAP_compute_res_release(ctx); return -4; }
  unsigned short* Av=(unsigned short*)(vb + 0);
  unsigned short* Wv=(unsigned short*)(vb + (size_t)MT*KT*2048);
  unsigned short* Cv=(unsigned short*)(vb + ((size_t)MT*KT + (size_t)KT*NT)*2048);
  int*            Bv=(int*)(vb + ((size_t)MT*KT + (size_t)KT*NT + (size_t)MT*NT)*2048);
  unsigned char* base=(unsigned char*)c;
  unsigned short* Ad=(unsigned short*)(base + 4096);
  unsigned short* Wd=(unsigned short*)(base + 4096 + (size_t)M*K*2);
  unsigned short* Cd=(unsigned short*)(base + 4096 + (size_t)M*K*2 + (size_t)K*N*2);
  memset(Bv,0,256); for(k=0;k<32;k++) Bv[k]=0x00003C00;     /* scale=1.0 col table */
  unsigned long long tp0=HAP_perf_get_time_us();
  for(mt=0;mt<MT;mt++) for(kt=0;kt<KT;kt++){ unsigned short* t=Av + ((size_t)mt*KT+kt)*1024;
    for(r=0;r<32;r++) for(k=0;k<32;k++) t[HMX_IDX(r,k)]=Ad[(size_t)(mt*32+r)*K + kt*32+k]; }
  for(nt=0;nt<NT;nt++) for(kt=0;kt<KT;kt++){ unsigned short* t=Wv + ((size_t)nt*KT+kt)*1024;
    for(k=0;k<32;k++) for(o=0;o<32;o++) t[HMX_IDX(k,o)]=Wd[(size_t)(kt*32+k)*N + nt*32+o]; }
  unsigned long long tp1=HAP_perf_get_time_us();
  compute_res_hmx_mutex_t mtx;
  if(HAP_compute_res_hmx_lock3(ctx,HAP_COMPUTE_RES_HMX_NON_SHARED,&mtx,1000000)){ HAP_compute_res_release(ctx); return -5; }
  unsigned int range=(unsigned)(KT*2048 - 1);
  int REP=(int)c[17]; if(REP<1) REP=1;                       /* repeat compute for stable timing (timer=1us) */
  unsigned long long tc0=HAP_perf_get_time_us();
  asm volatile("bias = mxmem(%0)\n" :: "r"((unsigned int)(unsigned long)Bv) : "memory");
  int rep; for(rep=0;rep<REP;rep++)
  for(mt=0;mt<MT;mt++) for(nt=0;nt<NT;nt++){
    unsigned int Aa=(unsigned int)(unsigned long)(Av + (size_t)mt*KT*1024);
    unsigned int Wa=(unsigned int)(unsigned long)(Wv + (size_t)nt*KT*1024);
    unsigned int Ca=(unsigned int)(unsigned long)(Cv + ((size_t)mt*NT+nt)*1024);
    asm volatile("mxclracc.hf\n");
    asm volatile("{\n activation.hf = mxmem(%1, %0):deep\n weight.hf = mxmem(%2, %0)\n}\n"
                 :: "r"(range), "r"(Aa), "r"(Wa));
    asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(Ca), "r"(0u) : "memory");
  }
  unsigned long long tc1=HAP_perf_get_time_us();
  HAP_compute_res_hmx_unlock3(ctx,HAP_COMPUTE_RES_HMX_NON_SHARED,&mtx);
  c[32]=(unsigned)REP;
  for(mt=0;mt<MT;mt++) for(nt=0;nt<NT;nt++){ unsigned short* t=Cv + ((size_t)mt*NT+nt)*1024;
    for(r=0;r<32;r++) for(o=0;o<32;o++) Cd[(size_t)(mt*32+r)*N + nt*32+o]=t[HMX_IDX(r,o)]; }
  c[30]=(unsigned)(tc1-tc0); c[31]=(unsigned)(tp1-tp0);
  HAP_compute_res_release(ctx);
  return 0;
}

/* 2-NSP HMX split GEMM: each NSP (role) does HMX on its half of the K-contraction (k0=role*K/2),
   produces a partial C[M,N] into shared DDR (P0/P1), then a cross-NSP all-reduce sums them -> Y.
   Both NSPs run this concurrently; they sync via hmx_barrier. role0 records per-iter wall-us to c[30],
   and the pure half-K HMX-compute us to c[33]. DDR: A@4096, W after A, P0,P1,Y after W. */
int hmx_split_bench(volatile unsigned int* c, unsigned int role){
  int M=(int)c[14], K=(int)c[15], N=(int)c[16]; int REP=(int)c[17]; if(REP<1) REP=1;
  int Khalf=K/2, k0=(int)role*Khalf;
  int MT=M/32, KTh=Khalf/32, NT=N/32, mt,kt,nt,r,k,o,rep;
  unsigned int phase=0; c[42+role]=1; c[44+role]=0;
  HAP_power_request_t preq; memset(&preq,0,sizeof(preq));
  preq.type=HAP_power_set_HMX; preq.hmx.power_up=1; HAP_power_set((void*)&g_pwr_anchor,&preq);
  compute_res_attr_t attr; if(compute_resource_attr_init(&attr)){ c[44+role]=1; return -1; }
  unsigned int vneed=(unsigned)(((size_t)MT*KTh + (size_t)KTh*NT + (size_t)MT*NT)*2048 + 2048);
  if(compute_resource_attr_set_vtcm_param(&attr, vneed, 0)) return -2;
  c[42+role]=2;
  unsigned int ctx=HAP_compute_res_acquire(&attr,1000000); if(!ctx){ c[44+role]=3; return -3; }
  c[42+role]=3;
  unsigned char* vb=(unsigned char*)compute_resource_attr_get_vtcm_ptr(&attr);
  if(!vb){ c[44+role]=4; HAP_compute_res_release(ctx); return -4; }
  c[42+role]=4;
  unsigned short* Av=(unsigned short*)(vb+0);
  unsigned short* Wv=(unsigned short*)(vb+(size_t)MT*KTh*2048);
  unsigned short* Cv=(unsigned short*)(vb+((size_t)MT*KTh+(size_t)KTh*NT)*2048);
  int*            Bv=(int*)(vb+((size_t)MT*KTh+(size_t)KTh*NT+(size_t)MT*NT)*2048);
  unsigned char* base=(unsigned char*)c;
  unsigned short* Ad=(unsigned short*)(base+4096);
  unsigned short* Wd=(unsigned short*)(base+4096+(size_t)M*K*2);
  unsigned short* P0=(unsigned short*)(base+4096+(size_t)M*K*2+(size_t)K*N*2);
  unsigned short* P1=(unsigned short*)((unsigned char*)P0+(size_t)M*N*2);
  unsigned short* Y =(unsigned short*)((unsigned char*)P1+(size_t)M*N*2);
  unsigned short* Pr=(role==0)?P0:P1;
  memset(Bv,0,256); for(k=0;k<32;k++) Bv[k]=0x00003C00;
  /* pack my K-half once */
  for(mt=0;mt<MT;mt++) for(kt=0;kt<KTh;kt++){ unsigned short* t=Av+((size_t)mt*KTh+kt)*1024;
    for(r=0;r<32;r++) for(k=0;k<32;k++) t[HMX_IDX(r,k)]=Ad[(size_t)(mt*32+r)*K + k0 + kt*32+k]; }
  for(nt=0;nt<NT;nt++) for(kt=0;kt<KTh;kt++){ unsigned short* t=Wv+((size_t)nt*KTh+kt)*1024;
    for(k=0;k<32;k++) for(o=0;o<32;o++) t[HMX_IDX(k,o)]=Wd[(size_t)(k0+kt*32+k)*N + nt*32+o]; }
  c[42+role]=5;
  compute_res_hmx_mutex_t mtx;
  if(HAP_compute_res_hmx_lock3(ctx,HAP_COMPUTE_RES_HMX_NON_SHARED,&mtx,1000000)){ c[44+role]=5; HAP_compute_res_release(ctx); return -5; }
  c[42+role]=6;
  unsigned int range=(unsigned)(KTh*2048-1);
  asm volatile("bias = mxmem(%0)\n" :: "r"((unsigned int)(unsigned long)Bv) : "memory");
  hmx_barrier(c, role, &phase);                  /* both packed + ready */
  c[42+role]=7;
  unsigned long long tcsum=0, t0=HAP_perf_get_time_us();
  for(rep=0;rep<REP;rep++){
    c[46+role]=(unsigned)rep; c[48+role]=1;
    unsigned long long tc0=HAP_perf_get_time_us();
    for(mt=0;mt<MT;mt++) for(nt=0;nt<NT;nt++){
      unsigned int Aa=(unsigned int)(unsigned long)(Av+(size_t)mt*KTh*1024);
      unsigned int Wa=(unsigned int)(unsigned long)(Wv+(size_t)nt*KTh*1024);
      unsigned int Ca=(unsigned int)(unsigned long)(Cv+((size_t)mt*NT+nt)*1024);
      asm volatile("mxclracc.hf\n");
      asm volatile("{\n activation.hf = mxmem(%1, %0):deep\n weight.hf = mxmem(%2, %0)\n}\n" :: "r"(range), "r"(Aa), "r"(Wa));
      asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(Ca), "r"(0u) : "memory");
    }
    tcsum += HAP_perf_get_time_us()-tc0; c[48+role]=2;
    for(mt=0;mt<MT;mt++) for(nt=0;nt<NT;nt++){ unsigned short* t=Cv+((size_t)mt*NT+nt)*1024;
      for(r=0;r<32;r++) for(o=0;o<32;o++) Pr[(size_t)(mt*32+r)*N + nt*32+o]=t[HMX_IDX(r,o)]; }
    c[48+role]=3;
    hmx_barrier(c, role, &phase);                /* both partials in DDR */
    c[48+role]=4;
    if(role==0){ int MN=M*N,i; for(i=0;i<MN;i++) Y[i]=f2h((float)h2f(P0[i])+(float)h2f(P1[i])); }  /* all-reduce */
    c[48+role]=5;
    hmx_barrier(c, role, &phase);                /* NSP1 waits for sum before next overwrite */
    c[48+role]=6;
  }
  if(role==0){ c[30]=(unsigned)((HAP_perf_get_time_us()-t0)/(unsigned)REP); c[33]=(unsigned)(tcsum/(unsigned)REP); }
  c[42+role]=100;
  HAP_compute_res_hmx_unlock3(ctx,HAP_COMPUTE_RES_HMX_NON_SHARED,&mtx);
  HAP_compute_res_release(ctx);
  return 0;
}