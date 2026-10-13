/* HVX FAST-9-16 corner kernel (uint8 vision "glue") -- the integer side of the Hexagon int/fp microbench.
   Our kernel, extracted on its own: drop it into a Hexagon SDK cDSP skeleton (e.g. the SDK's calculator example,
   which is Qualcomm-licensed and not redistributed here) and call hvx_fast_rows() from your skel method.
   Build for v73 with HVX (-mhvx -mhvx-length=128B). Needs <hexagon_types.h> and <hvx_hexagon_protos.h>. */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/* ---- HVX FAST-9-16 over a band of rows [y0,y1); returns corner count ---- */
static const int ox2[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
static const int oy2[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};

static unsigned int hvx_fast_rows(const unsigned char* img, unsigned int w, unsigned int hh,
                                  unsigned int barrier, unsigned int y0, unsigned int y1){
  const int VLEN=128; unsigned int total=0;
  HVX_Vector vt=Q6_Vb_vsplat_R((int)barrier), one=Q6_Vb_vsplat_R(1), z=Q6_V_vzero(), eight=Q6_Vb_vsplat_R(8);
  for(unsigned int y=y0;y<y1;y++){
    HVX_Vector accw=Q6_V_vzero(); unsigned int xlast=3;
    for(unsigned int x=3;x+VLEN<=w-3;x+=VLEN){
      const unsigned char* c=&img[y*w+x];
      HVX_Vector C=*(HVX_UVector*)c;
      HVX_Vector hi=Q6_Vub_vadd_VubVub_sat(C,vt), lo=Q6_Vub_vsub_VubVub_sat(C,vt);
      HVX_Vector runB=z,runD=z,maxB=z,maxD=z;
      for(int j=0;j<24;j++){ int k=j&15;
        HVX_Vector Vk=*(HVX_UVector*)&img[(y+oy2[k])*w + x+ox2[k]];
        HVX_VectorPred pB=Q6_Q_vcmp_gt_VubVub(Vk,hi), pD=Q6_Q_vcmp_gt_VubVub(lo,Vk);
        runB=Q6_V_vmux_QVV(pB,Q6_Vb_vadd_VbVb(runB,one),z);
        runD=Q6_V_vmux_QVV(pD,Q6_Vb_vadd_VbVb(runD,one),z);
        maxB=Q6_Vub_vmax_VubVub(maxB,runB); maxD=Q6_Vub_vmax_VubVub(maxD,runD);
      }
      HVX_VectorPred cB=Q6_Q_vcmp_gt_VubVub(maxB,eight), cD=Q6_Q_vcmp_gt_VubVub(maxD,eight);
      HVX_Vector corner=Q6_V_vmux_QVV(cB,one,Q6_V_vmux_QVV(cD,one,z));
      accw=Q6_Vuw_vrmpyacc_VuwVubRub(accw,corner,0x01010101u);
      xlast=x+VLEN;
    }
    unsigned int tmp[32]; *(HVX_UVector*)tmp=accw; for(int q=0;q<32;q++) total+=tmp[q];
    if(xlast < w-3){
      unsigned int xf=w-3-VLEN, M=xlast-xf;
      unsigned char mb[128]; for(int q=0;q<128;q++) mb[q]=(q>=(int)M)?1:0;
      HVX_Vector mask=*(HVX_UVector*)mb;
      HVX_Vector C=*(HVX_UVector*)&img[y*w+xf];
      HVX_Vector hi=Q6_Vub_vadd_VubVub_sat(C,vt), lo=Q6_Vub_vsub_VubVub_sat(C,vt);
      HVX_Vector runB=z,runD=z,maxB=z,maxD=z;
      for(int j=0;j<24;j++){ int k=j&15;
        HVX_Vector Vk=*(HVX_UVector*)&img[(y+oy2[k])*w + xf+ox2[k]];
        HVX_VectorPred pB=Q6_Q_vcmp_gt_VubVub(Vk,hi), pD=Q6_Q_vcmp_gt_VubVub(lo,Vk);
        runB=Q6_V_vmux_QVV(pB,Q6_Vb_vadd_VbVb(runB,one),z);
        runD=Q6_V_vmux_QVV(pD,Q6_Vb_vadd_VbVb(runD,one),z);
        maxB=Q6_Vub_vmax_VubVub(maxB,runB); maxD=Q6_Vub_vmax_VubVub(maxD,runD);
      }
      HVX_VectorPred cB=Q6_Q_vcmp_gt_VubVub(maxB,eight), cD=Q6_Q_vcmp_gt_VubVub(maxD,eight);
      HVX_Vector corner=Q6_V_vmux_QVV(cB,one,Q6_V_vmux_QVV(cD,one,z));
      corner=Q6_V_vand_VV(corner,mask);
      HVX_Vector a2=Q6_Vuw_vrmpy_VubRub(corner,0x01010101u);
      unsigned int t2[32]; *(HVX_UVector*)t2=a2; for(int q=0;q<32;q++) total+=t2[q];
    }
  }
  return total;
}

