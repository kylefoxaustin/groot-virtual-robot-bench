import sys, csv, collections
fn, stage = sys.argv[1], sys.argv[2]
def kb(n):
    s=n.lower()
    return 'matmul' if any(k in s for k in ['gemm','matmul','mha','conv','myl_fc','fc_','cutlass','implicit','wgrad','fprop']) else 'glue'
lines=[l for l in open(fn,errors='ignore') if l.lstrip().startswith('"')]   # ncu CSV rows are quoted; trtexec logs are not
if not lines: print(f"{stage}: NO CSV ROWS (ncu emitted none)"); sys.exit(0)
rd=list(csv.reader(lines)); hdr=rd[0]
def col(sub):
    for i,h in enumerate(hdr):
        if h.strip().endswith(sub): return i
    return None
ci={'int':col('op_integer_pred_on.sum'),'fp16':col('op_fp16_pred_on.sum'),
    'fp32':col('op_fp32_pred_on.sum'),'fp64':col('op_fp64_pred_on.sum'),'tensor':col('pipe_tensor.sum')}
ctot=next((i for i,h in enumerate(hdr) if h.strip().endswith('thread_inst_executed.sum') and 'op_' not in h and 'pipe' not in h), None)
ki=hdr.index('Kernel Name') if 'Kernel Name' in hdr else None
def num(x):
    try:
        v=float(str(x).replace(',','').strip() or 0)
        return v if (v==v and v not in (float('inf'),float('-inf'))) else 0.0
    except: return 0.0
agg=collections.Counter(); bk={'matmul':collections.Counter(),'glue':collections.Counter()}; nk=0
for r in rd[1:]:
    if len(r)<len(hdr) or r==hdr: continue
    b=kb(r[ki] if ki is not None else ''); nk+=1
    for tag,idx in ci.items():
        if idx is not None: v=num(r[idx]); agg[tag]+=v; bk[b][tag]+=v
    if ctot is not None: v=num(r[ctot]); agg['total']+=v; bk[b]['total']+=v
def rep(c,label):
    tot=c.get('total',0) or (c['int']+c['fp16']+c['fp32']+c['fp64']) or 1
    fp=c['fp16']+c['fp32']+c['fp64']; other=max(0.0,tot-c['int']-fp)
    print(f"  {label:7s} INT {100*c['int']/tot:5.1f}%  FP {100*fp/tot:5.1f}% (fp16 {100*c['fp16']/tot:.1f}/fp32 {100*c['fp32']/tot:.1f})  OTHER {100*other/tot:5.1f}%  tensor_inst={c['tensor']:.4g}  total_inst={tot:.4g}")
print(f"{stage}  (kernel-rows={nk}):"); rep(agg,'ALL'); rep(bk['matmul'],'matmul'); rep(bk['glue'],'glue')
