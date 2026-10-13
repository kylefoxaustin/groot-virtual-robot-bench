#!/usr/bin/env python3
"""Static instruction mix of a hot loop in a Hexagon object (int vs fp vs memory).

Disassembles OBJ with hexagon-llvm-objdump (--mcpu=hexagonv73), takes FUNCTION, picks the innermost
hardware loop (loop0 ... :endloop0) whose body contains MARKER, and classifies every instruction in it:
  FP   : any .hf / .sf / .qf operation, incl. HMX (mxclracc.hf, activation/weight.hf = mxmem, mxmem:after.hf)
         -- on HMX the matrix loads ARE the systolic multiply-accumulate, there is no separate multiply mnemonic
  MEM  : scalar / vector loads and stores (mem*, vmem*) that are not HMX mxmem
  INT  : everything else that does work (adds, compares, muxes, shifts, integer vector ops)
  skip : nop, loop/branch control
For a fixed-trip inner loop the static mix ~= the dynamic mix.
usage: disasm_mix.py OBJ FUNCTION MARKER [--objdump PATH]
       disasm_mix.py --text DISASM.txt FUNCTION MARKER      (re-classify a saved objdump listing; no SDK needed)
  e.g. disasm_mix.py calculator_hmx.o hmx_gemm mxclracc
       disasm_mix.py calculator_imp.o hvx_fast_rows vcmp"""
import os, re, subprocess, sys

argv = sys.argv[1:]
if "--text" in argv:
    i = argv.index("--text"); txt = open(argv[i + 1]).read(); del argv[i:i + 2]
    fn, marker = argv[0], argv[1]
else:
    od = argv[argv.index("--objdump") + 1] if "--objdump" in argv else os.environ.get("HEXAGON_OBJDUMP", "hexagon-llvm-objdump")
    a = [x for j, x in enumerate(argv) if not x.startswith("--") and (j == 0 or argv[j - 1] != "--objdump")]
    obj, fn, marker = a[0], a[1], a[2]
    txt = subprocess.run([od, "-d", "--mcpu=hexagonv73", obj], capture_output=True, text=True, check=True).stdout
body = txt.split(f"<{fn}>:", 1)[1].split("\n\n", 1)[0].splitlines()
ins = []
for ln in body:
    m = re.match(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s){4}\s*[0-9a-f]{8}\s+(.*)", ln)
    if m: ins.append((int(m.group(1), 16), m.group(2)))
# loops: "loop0(0xADDR,...)" sets the body start; the body ends at the packet carrying ":endloop0"
loops = []
for addr, s in ins:
    m = re.search(r"loop0\(0x([0-9a-f]+)", s)
    if m:
        start = int(m.group(1), 16)
        end = next((a2 for a2, s2 in ins if a2 >= start and ":endloop0" in s2), None)
        if end is not None: loops.append((start, end))
loop = next(((s0, e0) for s0, e0 in loops if any(marker in s for a2, s in ins if s0 <= a2 <= e0)), None)
if loop is None: sys.exit(f"no loop0 body containing {marker!r} in {fn}")


def cls(s):
    s = s.strip(" {}").replace(":endloop0", "").replace(":endloop1", "").strip()
    if not s or s.startswith(("nop", "loop", "jump", "if (", "call")) or "jump" in s: return None, s
    if re.search(r"\.(hf|sf|qf16|qf32)\b", s) or "mxclracc" in s: return "FP", s
    if re.search(r"\b(v?memu?[bhwd]?|vmemu?)\b|mem[bhwd]?\(|memu[bh]\(|vmem", s) and "mxmem" not in s: return "MEM", s
    return "INT", s


rows = [cls(s) + (a2,) for a2, s in ins if loop[0] <= a2 <= loop[1]]
cnt = {"FP": 0, "INT": 0, "MEM": 0}
for c, s, a2 in rows:
    print(f"  {a2:#06x}  {c or '-':4s} {s}")
    if c: cnt[c] += 1
tot = sum(cnt.values())
print(f"{fn} hot loop {loop[0]:#x}-{loop[1]:#x}: " + "  ".join(f"{k} {v} ({100*v/tot:.1f}%)" for k, v in cnt.items()))
