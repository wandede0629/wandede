r"""作者-年份 -> 数字标号 [n] (JPS/Vancouver)。默认干跑(打印每处替换预览);--apply 写回。"""
import re, sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "build_docx.js"
text = SRC.read_text(encoding="utf-8")
APPLY = "--apply" in sys.argv

m = re.search(r"const refs = \[(.*?)\n\];", text, re.S)
refs_block, refs_lo, refs_hi = m.group(1), m.start(), m.end()
ref_strs = re.findall(r'"((?:[^"\\]|\\.)*)"', refs_block)

def ref_key(s):
    sn = re.match(r"\s*([A-Za-zÀ-ÿ'’-]+)", s).group(1).lower()
    yr = re.search(r"\b(19|20)\d\d\b", s).group(0)
    return sn, yr

keys = [ref_key(s) for s in ref_strs]
key2idx = {}
for i, k in enumerate(keys):
    key2idx.setdefault(k, i)

body = text[:refs_lo]

def part_key(part):
    yr = re.search(r"\b(19|20)\d\d\b", part)
    sn = re.match(r"\s*([A-Za-zÀ-ÿ'’-]+)", part)
    if yr and sn and (sn.group(1).lower(), yr.group(0)) in key2idx:
        return (sn.group(1).lower(), yr.group(0))
    return None

NARR = re.compile(r"([A-Z][A-Za-zÀ-ÿ'’.-]+(?:\s+et al\.| and [A-Z][A-Za-zÀ-ÿ'’.-]+)?)\s*\((\d{4})[a-z]?\)")
PAREN = re.compile(r"\(([^()]*\b(?:19|20)\d\d[a-z]?\b[^()]*)\)")
CITE_IN = re.compile(r"[A-Za-zÀ-ÿ'’.-]+(?: et al\.| and [A-Za-zÀ-ÿ'’.-]+)?,\s*(?:19|20)\d\d")

# ---- 收集所有引用 span(含位置), 不重叠 ----
spans = []   # (start, end, kind, payload)
for mt in NARR.finditer(body):
    sn = re.match(r"([A-Za-zÀ-ÿ'’-]+)", mt.group(1)).group(1).lower()
    k = (sn, mt.group(2))
    if k in key2idx:
        spans.append((mt.start(), mt.end(), "narr", (mt.group(1), k)))
for mt in PAREN.finditer(body):
    inner = mt.group(1)
    if not CITE_IN.search(inner):
        continue
    parts = inner.split(";")
    pk = [part_key(p) for p in parts]
    if not any(pk):
        continue
    spans.append((mt.start(), mt.end(), "paren", list(zip(parts, pk))))

spans.sort()
# 去重叠(narr 的 (year) 不会落入 paren 引用; 仍校验)
clean = []
last = -1
for s in spans:
    if s[0] >= last:
        clean.append(s); last = s[1]
    else:
        print("⚠ 重叠 span 跳过:", body[s[0]:s[1]])
spans = clean

# ---- 按出现顺序编号 ----
key2num, order = {}, []
def assign(k):
    if k not in key2num:
        order.append(k); key2num[k] = len(order)
    return key2num[k]

for st, en, kind, pl in spans:
    if kind == "narr":
        assign(pl[1])
    else:
        for part, k in pl:
            if k:
                assign(k)

# ---- 构造替换 ----
def merge_adjacent(s):
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\[(\d+)\]\s*;\s*\[(\d+)\]", lambda mm: "[" + ",".join(str(x) for x in sorted({int(mm.group(1)), int(mm.group(2))})) + "]", s)
    return s

repls = []   # (start, end, new)
for st, en, kind, pl in spans:
    if kind == "narr":
        name, k = pl
        new = f"{name} [{key2num[k]}]"
    else:
        allnum = all(k for _, k in pl)
        if allnum:
            nums = sorted({key2num[k] for _, k in pl})
            new = "[" + ",".join(map(str, nums)) + "]"
        else:
            segs = []
            for part, k in pl:
                segs.append(f"[{key2num[k]}]" if k else part.strip())
            new = "(" + "; ".join(segs) + ")"
            new = merge_adjacent(new)
    repls.append((st, en, new))

# ---- 预览 ----
print(f"refs {len(ref_strs)} 篇; 引用 span {len(spans)} 处; 命中 {len(order)} 篇")
uncited = [i for i in range(len(ref_strs)) if keys[i] not in key2num]
print("未被引文献:", [f"#{i+1} {keys[i]}" for i in uncited] or "无 ✓")
print("\n=== 全部替换预览 ===")
for st, en, new in repls:
    print(f"  {body[st:en][:58]:60} -> {new}")

if not APPLY:
    print("\n(干跑; 加 --apply 写回)")
    sys.exit()

# ---- 写回: 正文替换(从后往前) + refs 重排 ----
newbody = body
for st, en, new in sorted(repls, reverse=True):
    newbody = newbody[:st] + new + newbody[en:]

new_refs = [ref_strs[key2idx[k]] for k in order]
def esc(s): return s.replace("\\", "\\\\").replace('"', '\\"')
refs_js = "const refs = [\n" + "".join(f'  "{esc(s)}",\n' for s in new_refs) + "]"
newtext = newbody + refs_js + text[refs_hi:]
SRC.write_text(newtext, encoding="utf-8")
print(f"\n✅ 已写回 build_docx.js ({len(new_refs)} 篇按出现顺序重排)")
