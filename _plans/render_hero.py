"""Side-by-side unmasking animation from recorded PIG decodes: Fast-dLLM (left) vs PIG.INT (right).
Frame k = canvas after k forward passes; committed tokens are tinted by the pass that committed them."""
import json, sys, colorsys
from PIL import Image, ImageDraw, ImageFont

traces = json.load(open(sys.argv[1])); name = sys.argv[2]; out = sys.argv[3]
COLS = int(sys.argv[4]) if len(sys.argv) > 4 else 58
TITLE = sys.argv[5] if len(sys.argv) > 5 else "Same model, same prompt, same output.  Which masked positions to commit in parallel?"
SUB = sys.argv[6] if len(sys.argv) > 6 else "LLaDA-8B-Instruct · HumanEval/2 · 64-token blocks · one frame = one forward pass"
SCALE = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANSB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F = ImageFont.truetype(MONO, 15); FS = ImageFont.truetype(SANS, 14); FB = ImageFont.truetype(SANSB, 17); FT = ImageFont.truetype(SANSB, 20); FSM = ImageFont.truetype(SANS, 12)
CW, CH = int(F.getlength("M")), 21

entry = traces[name]
lanes = [("fastdllm", "Fast-dLLM  (confidence threshold ν = 0.85)"), ("pig_int", "PIG.INT  (regret budget φ = 0.1)")]
n_max = max(entry["lanes"][l]["forwards"] for l, _ in lanes)

def layout(pieces, eos_at):
    """-> list over tokens of [(col,row,char)] cells for the text part, plus number of rows."""
    cells, col, row = [], 0, 0
    for i, piece in enumerate(pieces[:eos_at]):
        piece = piece.replace("\t", "    ")
        tc = []
        segs = piece.split("\n")
        for si, seg in enumerate(segs):
            if si > 0:
                tc.append((min(col, COLS - 1), row, "↵"))   # end-of-line marker for the newline itself
                row += 1; col = 0
            if seg and col + len(seg) > COLS and col > 0:
                row += 1; col = 0
            for ch in seg:
                if col >= COLS:
                    row += 1; col = 0
                tc.append((col, row, ch)); col += 1
        cells.append(tc)
    return cells, row + 1

L = {}
for lane, _ in lanes:
    d = entry["lanes"][lane]
    cells, rows = layout(d["pieces"], d["eos_at"])
    L[lane] = dict(cells=cells, rows=rows, round=d["round"], forwards=d["forwards"], eos_at=d["eos_at"], n=len(d["pieces"]))
ROWS = max(L[l]["rows"] for l, _ in lanes)
TAIL = max(L[l]["n"] - L[l]["eos_at"] for l, _ in lanes)
SQ, SQG = 7, 2                                   # padding-strip squares
TAIL_COLS = (COLS * CW) // (SQ + SQG)
TAIL_ROWS = (TAIL + TAIL_COLS - 1) // TAIL_COLS if TAIL else 0

PAD, GAP, HEAD = 22, 34, 140
PW = COLS * CW + 2 * 12
PH = ROWS * CH + 2 * 12
TAIL_H = (TAIL_ROWS * (SQ + SQG) + 26) if TAIL else 0
W = 2 * PW + GAP + 2 * PAD
H = HEAD + PH + TAIL_H + 70

def tint(k):                                     # pass index -> pastel colour, same scale in both panels
    h = 0.62 - 0.62 * (k / max(n_max - 1, 1))    # blue -> red as passes go by
    r, g, b = colorsys.hls_to_rgb(h, 0.80, 0.75)
    return (int(r * 255), int(g * 255), int(b * 255))
MASK, MASKTXT, INK, GREY = (222, 225, 230), (150, 155, 165), (20, 20, 20), (110, 110, 110)

def frame(k):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.text((PAD, 16), TITLE, font=FT, fill=INK)
    dr.text((PAD, 48), SUB, font=FS, fill=GREY)
    # legend, right-aligned on the subtitle row
    lx, ly = PAD, 72
    dr.text((lx, ly), "committed in pass", font=FSM, fill=GREY)
    for j in range(n_max):
        x0 = lx + 112 + j * 6
        dr.rectangle([x0, ly + 2, x0 + 6, ly + 14], fill=tint(j))
    dr.text((lx + 112, ly + 16), "1", font=FSM, fill=GREY); dr.text((lx + 112 + n_max * 6 - 8, ly + 16), str(n_max), font=FSM, fill=GREY)
    dr.rectangle([lx + 112 + n_max * 6 + 20, ly + 2, lx + 112 + n_max * 6 + 32, ly + 14], fill=MASK)
    dr.text((lx + 112 + n_max * 6 + 38, ly), "still masked", font=FSM, fill=GREY)
    for li, (lane, title) in enumerate(lanes):
        d = L[lane]; x0 = PAD + li * (PW + GAP); y0 = HEAD
        kk = min(k, d["forwards"])
        done = k >= d["forwards"]
        committed = sum(1 for r in d["round"] if 0 <= r < kk)
        dr.text((x0, y0 - 30), title, font=FB, fill=INK)
        dr.rounded_rectangle([x0, y0, x0 + PW, y0 + PH], radius=8, outline=(225, 225, 225), fill=(252, 252, 253))
        for ti, tc in enumerate(d["cells"]):
            r = d["round"][ti]
            on = 0 <= r < kk
            for (c, row, ch) in tc:
                cx, cy = x0 + 12 + c * CW, y0 + 12 + row * CH
                dr.rectangle([cx, cy + 1, cx + CW, cy + CH - 1], fill=tint(r) if on else MASK)
                if on and ch != "↵":
                    dr.text((cx, cy + 2), ch, font=F, fill=INK)
        # padding strip (positions after the end-of-text token)
        if TAIL:
            ty = y0 + PH + 8
            dr.text((x0, ty), f"after end-of-text: {d['n'] - d['eos_at']} padding positions (still cost passes)", font=FSM, fill=GREY)
            for j in range(d["n"] - d["eos_at"]):
                r = d["round"][d["eos_at"] + j]
                on = 0 <= r < kk
                sx = x0 + 12 + (j % TAIL_COLS) * (SQ + SQG); sy = ty + 18 + (j // TAIL_COLS) * (SQ + SQG)
                dr.rectangle([sx, sy, sx + SQ, sy + SQ], fill=tint(r) if on else MASK)
        # counters
        cy = y0 + PH + TAIL_H + 10
        dr.text((x0, cy), f"forward passes: {kk:>2}     tokens committed: {committed:>3} / {d['n']}", font=FS, fill=INK)
        if done:
            dr.rounded_rectangle([x0 + PW - 150, cy - 4, x0 + PW, cy + 20], radius=6, fill=(230, 245, 232))
            dr.text((x0 + PW - 142, cy), f"done in {d['forwards']} passes ✓", font=FS, fill=(30, 110, 50))
    return im

frames, durs = [], []
for k in range(0, n_max + 1):
    frames.append(frame(k)); durs.append(700 if k == 0 else 170)
durs[-1] = 2600
if SCALE != 1.0:
    frames = [f.resize((int(W * SCALE), int(H * SCALE)), Image.LANCZOS) for f in frames]
frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
frames[-1].save(out.replace(".gif", "_last.png")); frames[min(9, n_max)].save(out.replace(".gif", "_mid.png"))
import os; print(f"{out}: {len(frames)} frames, {W}x{H}, {os.path.getsize(out)/1e6:.2f} MB; rows {ROWS}, tail {TAIL}")
