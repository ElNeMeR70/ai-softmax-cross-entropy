"""Stable Softmax & Cross-Entropy Loss"""
import math

def stable_softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(probs, target_idx):
    return -math.log(max(probs[target_idx], 1e-12))

if __name__ == "__main__":
    print("[RUN] Stable Softmax & Cross-Entropy...")
    probs = stable_softmax([1000.0, 1002.0, 1001.0])
    loss = cross_entropy(probs, target_idx=1)
    print(f"[OK] Softmax Probs: {[round(p, 4) for p in probs]} | Loss: {loss:.4f}")
