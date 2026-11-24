"""
Shared training + evaluation loop, so every experiment runs through identical code.

An "objective" is a function (model, x, y) -> (loss, predictions as 9-way class indices).
Swapping the objective is the ONLY difference between the 9-way and factorized runs.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score


# ---------- objectives ----------

def ce_9way(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.float(), y), logits.argmax(1)


def ce_factorized(model, x, y):
    """Two 3-way questions. class = 3*v + h, so h = y % 3 and v = y // 3."""
    h_logits, v_logits = model(x)
    loss = F.cross_entropy(h_logits.float(), y % 3) + F.cross_entropy(v_logits.float(), y // 3)
    pred = 3 * v_logits.argmax(1) + h_logits.argmax(1)
    return loss, pred


# ---------- metrics ----------

def scores(p, y):
    """Overall scores, plus how often each axis (horizontal / vertical) is right."""
    p, y = np.asarray(p), np.asarray(y)
    h_ok, v_ok = (p % 3) == (y % 3), (p // 3) == (y // 3)
    return {"acc": float((p == y).mean()),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(9), zero_division=0)),
            "bal_acc": float(balanced_accuracy_score(y, p)),
            "h_acc": float(h_ok.mean()), "v_acc": float(v_ok.mean())}


def evaluate(model, loader, objective, dev, use_amp):
    model.eval()
    preds, labels, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                loss, pred = objective(model, x, y)
            loss_sum += loss.item() * len(y)
            n += len(y)
            preds.append(pred.cpu()); labels.append(y.cpu())
    return loss_sum / n, torch.cat(preds).numpy(), torch.cat(labels).numpy()


# ---------- training ----------

def train(model, objective, train_loader, val_loader, ckpt_dir, epochs, lr, weight_decay, dev, use_amp,
          log_every=200):
    """Train with AdamW + cosine schedule. Saves best.pth (by val macro-F1) and last.pth (to resume)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    last_path, best_path = os.path.join(ckpt_dir, "last.pth"), os.path.join(ckpt_dir, "best.pth")

    start, history, best_f1 = 0, [], -1.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, history, best_f1 = ck["epoch"], ck["history"], ck["best_f1"]
        print(f"Resuming after epoch {start} (best val macro-F1 so far {best_f1:.3f})")

    for epoch in range(start, epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                loss, _ = objective(model, x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            if i % log_every == 0:
                print(f"  epoch {epoch + 1} batch {i}/{len(train_loader)}  loss {run_loss / i:.3f}  ({time.time() - t0:.0f}s)")
        sched.step()

        val_loss, vp, vy = evaluate(model, val_loader, objective, dev, use_amp)
        s = scores(vp, vy)
        row = {"epoch": epoch + 1, "train_loss": run_loss / len(train_loader), "val_loss": val_loss, **s,
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        improved = s["macro_f1"] > best_f1
        if improved:
            best_f1 = s["macro_f1"]
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history, "best_f1": best_f1},
                   last_path)
        print(f"epoch {epoch + 1:2d}/{epochs} | train loss {row['train_loss']:.3f} | val loss {val_loss:.3f} | "
              f"acc {s['acc']:.3f}  macro-F1 {s['macro_f1']:.3f}  H {s['h_acc']:.3f}  V {s['v_acc']:.3f} | "
              f"{row['minutes']:.1f} min" + ("  <- best, saved" if improved else ""))
    return history, best_f1, best_path
