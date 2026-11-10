#!/usr/bin/env python
"""Fine-tune the Auto-AVSR LRS3_V_WER19.1 lip reader on mouth-crop data.

  vendor/venv/bin/python -I train/finetune.py --data data/pretrain --epochs 5 \
      --unfreeze lipflow --out train/ckpt/lipflow5.pth

Data contract: <data>/manifest.jsonl lines {id, text, src, dur, mode}; <data>/crops/<id>.npy uint8 [T,96,96]
grayscale, 25 fps mouth crops (same crops the chaplin pipeline feeds the model before its 88px center crop);
text UPPERCASE, no punctuation. 5% of the *videos* (grouped by `src`) are held out for val WER.

--unfreeze: lipflow = front end + embed + encoder layer 0 (vendor/lipflow/train_vsr.py's scope)
            encoder = the whole encoder     all = encoder + decoder + CTC head
Loss = 0.1 CTC + 0.9 label-smoothed attention (the model's own objective). BatchNorm stats stay frozen.

Output: only the changed tensors (default) or, with --save full, the whole state dict. eval/eval.py
--checkpoint accepts both (it merges a partial file onto the baseline weights).
Val WER is attention-greedy (no beam, no LM): a fast proxy, lower is better, not comparable to beam WER.
"""
import argparse, json, math, os, random, re, sys, time
from argparse import Namespace
from pathlib import Path

import torch
torch.zeros(1, device="cuda")  # init CUDA before heavy imports (else OOM on GB10)
import numpy as np
import torch.nn.functional as F

WS = Path(__file__).resolve().parent.parent
CHAPLIN = WS / "vendor" / "chaplin"
BASE = CHAPLIN / "benchmarks/LRS3/models/LRS3_V_WER19.1"
SPM = WS / "vendor/auto_avsr/spm/unigram/unigram5000.model"
MEAN, STD = 0.421, 0.165
sys.path.insert(0, str(CHAPLIN))


def patch_bf16_attention():
    """Vendor attention builds its mask fill via numpy, which has no bfloat16; same math with torch.finfo."""
    from espnet.nets.pytorch_backend.transformer import attention as A

    def forward_attention(self, value, scores, mask, rtn_attn=False):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(self.attn), value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return (self.linear_out(x), self.attn) if rtn_attn else self.linear_out(x)

    A.MultiHeadedAttention.forward_attention = forward_attention


def load_model(dev):
    patch_bf16_attention()
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    confs = json.load(open(BASE / "model.json"))
    args = Namespace(**(confs if isinstance(confs, dict) else confs[2]))
    units = (CHAPLIN / "pipelines/tokens/unigram5000_units.txt").read_text().splitlines()
    token_list = ["<blank>"] + [u.split()[0] for u in units] + ["<eos>"]
    m = E2E(len(token_list), args)
    m.load_state_dict(torch.load(BASE / "model.pth", map_location="cpu"))
    return m.to(dev), token_list


class Tok:
    def __init__(self, token_list):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(SPM))
        self.id = {t: i for i, t in enumerate(token_list)}
        self.unk = self.id["<unk>"]

    def __call__(self, text):
        clean = " ".join(re.sub(r"[^A-Za-z' ]+", " ", text).upper().split())
        return [self.id.get(p, self.unk) for p in self.sp.encode(clean, out_type=str)]


def set_trainable(m, scope):
    for p in m.parameters():
        p.requires_grad_(False)
    mods = {"lipflow": [m.encoder.frontend, m.encoder.embed, m.encoder.encoders[0]],
            "encoder": [m.encoder],
            "all": [m.encoder, m.decoder, m.ctc]}[scope]
    for mod in mods:
        for p in mod.parameters():
            p.requires_grad_(True)
    return [n for n, p in m.named_parameters() if p.requires_grad]


def freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()


# ---------------- data ----------------
def load_manifest(data):
    rows = []
    for l in open(Path(data) / "manifest.jsonl"):
        if l.strip():
            r = json.loads(l)
            if (Path(data) / "crops" / f"{r['id']}.npy").is_file() and " ".join(r["text"].split()):
                rows.append(r)
    return rows


def split_by_video(rows, frac=0.05, seed=0):
    vids = sorted({r.get("src") or r["id"].rsplit("_", 1)[0] for r in rows})
    random.Random(seed).shuffle(vids)
    nval = max(1, math.ceil(len(vids) * frac)) if len(vids) > 1 else 0
    val = set(vids[:nval])
    key = lambda r: r.get("src") or r["id"].rsplit("_", 1)[0]
    return [r for r in rows if key(r) not in val], [r for r in rows if key(r) in val]


class Clips(torch.utils.data.Dataset):
    def __init__(self, data, rows, tok, train):
        self.dir, self.rows, self.train = Path(data) / "crops", rows, train
        self.ys = [tok(r["text"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(np.load(self.dir / f"{self.rows[i]['id']}.npy")).float() / 255.0  # [T,96,96]
        if self.train:
            a, b = random.randint(0, 8), random.randint(0, 8)
            x = x[:, a:a + 88, b:b + 88]
            if random.random() < 0.5:
                x = x.flip(-1)
            x = x * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06)
            T = x.shape[0]
            for _ in range(2):  # time masks up to 0.4 s
                w = random.randint(0, min(10, T // 5)); s = random.randint(0, max(T - w, 0))
                x[s:s + w] = x.mean()
        else:
            x = x[:, 4:92, 4:92]
        return (x - MEAN) / STD, self.ys[i]


def collate(items):
    B, T = len(items), max(x.shape[0] for x, _ in items)
    x = torch.zeros(B, 1, T, 88, 88)
    xl = torch.tensor([it[0].shape[0] for it in items])
    for i, (v, _) in enumerate(items):
        x[i, 0, :v.shape[0]] = v
    return x, xl, [y for _, y in items]


def make_batches(ds, max_frames, shuffle, rng):
    """Length-bucketed batches: sort by length (noisy when shuffling), cap padded frames per batch."""
    lens = [np.load(ds.dir / f"{r['id']}.npy", mmap_mode="r").shape[0] for r in ds.rows] if not hasattr(ds, "lens") else ds.lens
    ds.lens = lens
    order = sorted(range(len(ds)), key=lambda i: lens[i] + (rng.uniform(0, 15) if shuffle else 0))
    batches, cur, mx = [], [], 0
    for i in order:
        if cur and max(mx, lens[i]) * (len(cur) + 1) > max_frames:
            batches.append(cur); cur, mx = [], 0
        cur.append(i); mx = max(mx, lens[i])
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


# ---------------- model passes ----------------
def encode(m, x, xl, dev):
    x, xl = x.to(dev, non_blocking=True), xl.to(dev)
    T = x.shape[2]
    hmask = (torch.arange(T, device=dev)[None] < xl[:, None]).unsqueeze(1)  # [B,1,T]
    hs, _ = m.encoder(x, hmask)
    return hs, hmask, xl


def loss_fn(m, x, xl, ys, dev):
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    hs, hmask, xl = encode(m, x, xl, dev)
    B, eos = len(ys), m.eos
    logp = m.ctc.ctc_lo(hs).float().log_softmax(-1).transpose(0, 1)
    tl = torch.tensor([len(y) for y in ys], device=dev)
    ctc = F.ctc_loss(logp, torch.tensor([t for y in ys for t in y], device=dev), xl, tl,
                     blank=0, reduction="sum", zero_infinity=True)
    L = max(len(y) for y in ys) + 1
    ys_in = torch.full((B, L), eos, dtype=torch.long); ys_out = torch.full((B, L), -1, dtype=torch.long)
    for i, y in enumerate(ys):
        ys_in[i, :len(y) + 1] = torch.tensor([eos] + y); ys_out[i, :len(y) + 1] = torch.tensor(y + [eos])
    ys_in, ys_out = ys_in.to(dev), ys_out.to(dev)
    ymask = (ys_out != -1).unsqueeze(1) & subsequent_mask(L, device=dev).unsqueeze(0)
    pred, _ = m.decoder(ys_in, ymask, hs, hmask)
    att = m.criterion(pred.float(), ys_out) * B  # criterion divides by B (length_normalized_loss=0); undo
    ntok = int(tl.sum()) + B
    return (0.1 * ctc + 0.9 * att) / ntok


@torch.no_grad()
def greedy_decode(m, x, xl, dev, token_list):
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    hs, hmask, _ = encode(m, x, xl, dev)
    B, eos = hs.shape[0], m.eos
    ys = torch.full((B, 1), eos, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(min(150, hs.shape[1] // 3 + 5)):
        L = ys.shape[1]
        pred, _ = m.decoder(ys, subsequent_mask(L, device=dev).unsqueeze(0), hs, hmask)
        nxt = pred[:, -1].argmax(-1).masked_fill(done, eos)
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= nxt == eos
        if done.all():
            break
    out = []
    for row in ys[:, 1:].tolist():
        row = row[:row.index(eos)] if eos in row else row
        out.append("".join(token_list[t] for t in row).replace("▁", " ").strip())
    return out


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.upper().replace("'", "")).split())


def val_wer(m, ds, args, dev, token_list):
    import editdistance
    m.eval()
    errs = words = 0
    for idx in make_batches(ds, args.max_frames, False, random.Random(0)):
        x, xl, _ = collate([ds[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            hyps = greedy_decode(m, x, xl, dev, token_list)
        for i, h in zip(idx, hyps):
            ref = norm(ds.rows[i]["text"]).split()
            errs += editdistance.eval(ref, norm(h).split()); words += len(ref)
    return errs / max(words, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--unfreeze", choices=["lipflow", "encoder", "all"], default="lipflow")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-frames", type=int, default=1500, help="padded frames per micro-batch (25 fps)")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--patience", type=int, default=3, help="stop after this many epochs without val improvement")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-max", type=int, default=300, help="cap on val segments per epoch")
    ap.add_argument("--save", choices=["delta", "full"], default="delta")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda:0")

    m, token_list = load_model(dev)
    tok = Tok(token_list)
    rows = load_manifest(args.data)
    tr_rows, va_rows = split_by_video(rows, args.val_frac, args.seed)
    random.Random(args.seed).shuffle(va_rows); va_rows = va_rows[:args.val_max]
    tr, va = Clips(args.data, tr_rows, tok, True), Clips(args.data, va_rows, tok, False)
    print(f"{len(tr)} train / {len(va)} val segments ({len({r.get('src') for r in va_rows})} val videos)", flush=True)
    assert len(tr), "no training data"

    names = set_trainable(m, args.unfreeze)
    params = [p for p in m.parameters() if p.requires_grad]
    print(f"unfreeze={args.unfreeze}: {sum(p.numel() for p in params)/1e6:.1f}M trainable params", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    n_mb = len(make_batches(tr, args.max_frames, False, random.Random(0)))
    total = max(1, args.epochs * math.ceil(n_mb / args.accum)); warm = min(50, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    def save(state):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        full = m.state_dict()
        keep = full.keys() if args.save == "full" else names
        torch.save({k: (state[k] if k in state else full[k]).detach().cpu().clone() for k in keep}, args.out)

    best, bad, rng = None, 0, random.Random(args.seed)
    if len(va):
        print(f"epoch 0 (baseline): val WER {val_wer(m, va, args, dev, token_list)*100:.1f}%", flush=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); m.train(); freeze_bn(m)
        batches = make_batches(tr, args.max_frames, True, rng)
        dl = torch.utils.data.DataLoader(tr, batch_sampler=batches, collate_fn=collate, num_workers=args.workers,
                                         prefetch_factor=2 if args.workers else None, persistent_workers=False)
        tot, n = 0.0, 0
        opt.zero_grad()
        for k, (x, xl, ys) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss = loss_fn(m, x, xl, ys, dev)
            (loss / args.accum).backward()
            tot += float(loss.detach()); n += 1
            if (k + 1) % args.accum == 0 or k + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step(); sched.step(); opt.zero_grad()
        wer = val_wer(m, va, args, dev, token_list) if len(va) else float("nan")
        print(f"epoch {ep}/{args.epochs}: loss {tot/n:.3f}  val WER {wer*100:.1f}%  {time.time()-t0:.0f}s  "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
        if best is None or not (wer >= best):  # nan (no val) always saves the latest
            best, bad = wer, 0
            save(m.state_dict()); print(f"  saved {args.out}", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop: val WER not improving", flush=True); break
    print("done; best val WER", f"{best*100:.1f}%" if best == best else "n/a")


if __name__ == "__main__":
    main()
