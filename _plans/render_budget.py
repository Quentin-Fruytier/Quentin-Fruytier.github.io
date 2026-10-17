"""PIG.INT budget walk (GIF) and the D-hat heat-map (PNG) for one recorded round of a real decode."""
import sys, json, colorsys, numpy as np, torch, pig
from pig.rules import Round, confidence_order, _column_influence
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

out_dir, name, phi = sys.argv[1], sys.argv[2], 0.1
rec = torch.load("/ssd1/qdf76/pig_release_check/tf5/llada_branch_matrix.pt", weights_only=False)
table = pig.load_tables("llada")["confidence"]
traces = json.load(open(f"{out_dir}/llada_block64_traces.json"))[name]["lanes"]["pig_int"]
p = rec["prompts"][f"block64/mode=pig.int/phi=0.1|{name}"]
pieces = traces["pieces"]

def walk(r):
    cand, conf, D = [int(c) for c in r["cand"]], r["conf"], r["D"]
    order = confidence_order(cand, conf)
    idx = np.asarray(order, dtype=np.int64)
    mu = _column_influence(D, idx, order[0])
    rows, spent, S = [], 0.0, [order[0]]
    rows.append(dict(pos=order[0], conf=float(conf[order[0]]), mu=0.0, b=0.0, spent=0.0, admitted=True))
    stop = None
    for k in range(1, len(order)):
        c = order[k]
        b = table(float(conf[c]), float(mu[k]))
        if stop is None and spent + b > phi:
            stop = k
        admitted = stop is None
        rows.append(dict(pos=c, conf=float(conf[c]), mu=float(mu[k]), b=b, spent=spent + b if admitted else spent, admitted=admitted))
        if admitted:
            S.append(c); spent += b; mu += _column_influence(D, idx, c)
    return order, rows, S, stop

# pick a round with a clear stop: many admitted, some rejected, within the first block
best = None
for ri, r in enumerate(p["rounds"]):
    order, rows, S, stop = walk(r)
    if stop is None or len(order) < 12:
        continue
    score = len(S) if 4 <= len(S) <= 12 else -1
    if best is None or score > best[0]:
        best = (score, ri, order, rows, S, stop, r)
score, ri, order, rows, S, stop, r = best
print(f"round {ri}: {len(order)} candidates, {len(S)} admitted, stop at rank {stop}")

# ---------- budget walk GIF ----------
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; SANSB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
F, FB, FT, FM, FS = ImageFont.truetype(SANS, 14), ImageFont.truetype(SANSB, 14), ImageFont.truetype(SANSB, 20), ImageFont.truetype(MONO, 14), ImageFont.truetype(SANS, 12)
NSHOW = min(len(rows), stop + 4, 16)
W, H = 980, 150 + NSHOW * 30 + 40
INK, GREY, GREEN, RED, BLUE = (20, 20, 20), (120, 120, 120), (46, 150, 80), (205, 60, 50), (23, 114, 208)
def tok(pos):
    s = pieces[pos].replace("\n", "↵").replace(" ", "␣")
    return s if len(s) <= 10 else s[:9] + "…"
def frame(k):   # k = number of candidates examined so far (0..NSHOW)
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.text((22, 14), f"PIG.INT walks the candidates by confidence and spends a regret budget φ = {phi}", font=FT, fill=INK)
    dr.text((22, 44), f"LLaDA-8B-Instruct · HumanEval/2 · forward pass {ri + 1} · {len(order)} masked candidates in the block · p̂ = calibrated P(parallel token ≠ sequential token)", font=FS, fill=GREY)
    # budget bar
    bx, by, bw, bh = 22, 74, 640, 22
    dr.text((bx, by - 2), "", font=F, fill=INK)
    dr.rounded_rectangle([bx, by, bx + bw, by + bh], radius=6, outline=(200, 200, 200), fill=(245, 246, 248))
    spent = rows[min(k, len(rows)) - 1]["spent"] if k > 0 else 0.0
    spent_shown = min(spent, phi) if (k == 0 or k - 1 < stop) else spent
    fill_w = int(bw * min(spent_shown / phi, 1.0))
    dr.rounded_rectangle([bx, by, bx + max(fill_w, 6), by + bh], radius=6, fill=(140, 200, 160))
    over = k > stop
    if over:
        dr.rounded_rectangle([bx + bw - 6, by, bx + bw + 24, by + bh], radius=6, fill=(240, 170, 160))
    dr.text((bx + bw + 34, by + 3), f"Σ p̂ = {spent:.3f}" + (f"  >  φ = {phi}   → stop" if over else f"   of φ = {phi}"), font=FB, fill=RED if over else INK)
    dr.text((bx, by + bh + 4), "0", font=FS, fill=GREY); dr.text((bx + bw - 24, by + bh + 4), f"φ = {phi}", font=FS, fill=GREY)
    # table header
    y0 = 136
    cols = [(22, "rank"), (70, "position"), (150, "token"), (270, "confidence"), (470, "μ̂ (influence of admitted)"), (690, "p̂"), (770, "decision")]
    for x, t in cols: dr.text((x, y0), t, font=FB, fill=GREY)
    dr.line([22, y0 + 20, W - 22, y0 + 20], fill=(220, 220, 220))
    for i in range(NSHOW):
        rw = rows[i]; y = y0 + 28 + i * 30
        seen = i < k
        col = INK if seen else (200, 200, 200)
        dr.text((22, y), f"{i + 1}", font=FM, fill=col)
        dr.text((70, y), f"{rw['pos']:3d}", font=FM, fill=col)
        dr.text((150, y), tok(rw["pos"]), font=FM, fill=col)
        # confidence bar
        dr.rectangle([270, y + 3, 270 + 150, y + 15], fill=(240, 240, 240))
        dr.rectangle([270, y + 3, 270 + int(150 * rw["conf"]), y + 15], fill=(170, 190, 230) if seen else (235, 235, 235))
        dr.text((428, y), f"{rw['conf']:.2f}", font=FM, fill=col)
        mu_w = int(min(rw["mu"], 3.0) / 3.0 * 150)
        dr.rectangle([470, y + 3, 470 + 150, y + 15], fill=(240, 240, 240))
        dr.rectangle([470, y + 3, 470 + mu_w, y + 15], fill=(235, 200, 150) if seen else (235, 235, 235))
        dr.text((628, y), f"{rw['mu']:.2f}", font=FM, fill=col)
        dr.text((690, y), f"{rw['b']:.3f}" if i > 0 else "first", font=FM, fill=col)
        if seen:
            if rw["admitted"]:
                dr.text((770, y), "commit now ✓", font=FB, fill=GREEN)
            elif i == stop:
                dr.text((770, y), "budget exceeded → wait", font=FB, fill=RED)
            else:
                dr.text((770, y), "wait (next pass)", font=F, fill=GREY)
    if k > stop:
        dr.text((22, H - 30), f"Committed this pass: {len(S)} tokens. The rest stay masked and get a fresh prediction next pass, now conditioned on these {len(S)}.", font=F, fill=INK)
    return im
frames = [frame(k) for k in range(0, stop + 2)] + [frame(stop + 1)] * 3
durs = [900] + [650] * (stop) + [1200] + [1200] * 3
frames[0].save(f"{out_dir}/budget-walk.gif", save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
frames[-1].save(f"{out_dir}/budget-walk_last.png")

# ---------- D-hat heat-map ----------
K = min(24, len(order))
sel = np.asarray(order[:K], dtype=np.int64)
D = np.asarray(r["D"], dtype=np.float64)[np.ix_(sel, sel)]
labels = [f"{tok(int(pp))}" for pp in sel]
fig, ax = plt.subplots(figsize=(10.2, 8.2), dpi=130)
v = np.percentile(np.abs(D), 98)
im = ax.imshow(D, cmap="RdBu", vmin=-v, vmax=v)   # blue: j makes i more certain (agreement); red: conflict
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels(labels, rotation=90, fontsize=8, family="DejaVu Sans Mono"); ax.set_yticklabels(labels, fontsize=8, family="DejaVu Sans Mono")
ax.set_xlabel("j: candidate committed (ordered by confidence, most confident first)", fontsize=10)
ax.set_ylabel("i: position whose prediction changes", fontsize=10)
ax.set_title(f"$\\hat{{D}}_{{ij}}$ = predicted entropy change at i if j were committed\nLLaDA-8B-Instruct · HumanEval/2 · forward pass {ri + 1} · top {K} of {len(order)} candidates", fontsize=11)
for s in range(len(S)):
    ax.add_patch(plt.Rectangle((-0.5, s - 0.5), K, 1, fill=False, edgecolor="none"))
ax.axhline(len(S) - 0.5, color="k", lw=1, ls="--"); ax.axvline(len(S) - 0.5, color="k", lw=1, ls="--")
ax.text(K - 0.7, len(S) - 1.0, f"committed this pass ({len(S)}) ↑", fontsize=9, ha="right", va="center", bbox=dict(fc="white", ec="none", alpha=0.85))
ax.text(K - 0.7, len(S) + 0.1, "↓ waiting for the next pass", fontsize=9, ha="right", va="center", bbox=dict(fc="white", ec="none", alpha=0.85))
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("nats   (blue: committing j makes i more certain — agreement;  red: less certain — conflict)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out_dir}/dhat-heatmap.png"); print("wrote dhat-heatmap.png", D.shape, f"v={v:.3f}")
