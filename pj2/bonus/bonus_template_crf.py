"""Bonus: 基于 template_for_crf.utf8 的手写线性链 CRF（仅中文）。

- 解析模板：U00..U09 一元、B00..B09 二元；`%x[row,col]` 取相对位置 token (col 始终 0)
- 一元特征（unigram）：作用于每个位置 → emission 累加；权重 W_U[f, t]
- 二元特征（bigram）：作用于每个 (k-1,k) 转移位置 → 位置特定 transition；权重 W_B[f, t_prev, t_cur]
- 训练：随机梯度法（SGD on log-linear CRF NLL）+ L2 正则
       loss = -log p(y | x; W) = log Z - score(gold)
       梯度: 期望特征频次 (forward-backward) - 经验特征频次
- 解码：位置-依赖 Viterbi（trans 是逐位置矩阵）

频次截断：unigram ≥ 2，bigram ≥ 5（控制规模）。

运行：
    cd bonus
    python bonus_template_crf.py            # 训练
    python bonus_template_crf_test.py       # 推断 + 评测
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import argparse
import math
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PJ_ROOT = os.path.dirname(HERE)
if PJ_ROOT not in sys.path:
    sys.path.insert(0, PJ_ROOT)

from data_utils import LABELS_CHN, load_corpus, ner_path  # noqa: E402

TEMPLATE_PATH = os.path.join(PJ_ROOT, "NER", "template_for_crf.utf8")


# ---------------- 模板解析 ----------------
def parse_template(path: str) -> Tuple[List[Tuple[str, List[Tuple[int, int]]]], List[Tuple[str, List[Tuple[int, int]]]]]:
    """返回 (unigram_templates, bigram_templates)。
    每个模板是 (prefix, [(row, col), ...])；prefix 形如 'U00'。
    """
    import re
    pat = re.compile(r"%x\[(-?\d+),(\d+)\]")
    unigrams, bigrams = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head, _, body = line.partition(":")
            offsets = [(int(r), int(c)) for r, c in pat.findall(body)]
            if head.startswith("U"):
                unigrams.append((head, offsets))
            elif head.startswith("B"):
                bigrams.append((head, offsets))
    return unigrams, bigrams


def expand_features(tokens: List[str], k: int,
                    templates: List[Tuple[str, List[Tuple[int, int]]]]) -> List[str]:
    """对位置 k 展开模板，返回特征字符串列表。
    超出边界时用 _B-i / _Ai 之类的边界占位（CRF++ 风格）。
    """
    L = len(tokens)
    out = []
    for prefix, offs in templates:
        parts = []
        for r, c in offs:
            j = k + r
            if j < 0:
                parts.append(f"_B{-j}")
            elif j >= L:
                parts.append(f"_A{j - L + 1}")
            else:
                parts.append(tokens[j])
        out.append(prefix + ":" + "/".join(parts))
    return out


# ---------------- 特征字典构建（单趟训练扫描） ----------------
def build_feature_index(train_sents, U_tpl, B_tpl,
                        min_freq_u: int = 2, min_freq_b: int = 5):
    u_cnt: Counter = Counter()
    b_cnt: Counter = Counter()
    for sent in train_sents:
        toks = [w for w, _ in sent]
        L = len(toks)
        for k in range(L):
            for f in expand_features(toks, k, U_tpl):
                u_cnt[f] += 1
        for k in range(1, L):
            for f in expand_features(toks, k, B_tpl):
                b_cnt[f] += 1
    u2id: Dict[str, int] = {}
    for f, c in u_cnt.items():
        if c >= min_freq_u:
            u2id[f] = len(u2id)
    b2id: Dict[str, int] = {}
    for f, c in b_cnt.items():
        if c >= min_freq_b:
            b2id[f] = len(b2id)
    return u2id, b2id


def precompute_sent_features(toks, U_tpl, B_tpl, u2id, b2id):
    """返回:
        u_feats: List[List[int]]  长度 L, 每个位置触发的 unigram 特征 id 列表
        b_feats: List[List[int]]  长度 L (k=0 为空), 每个 (k-1,k) 转移触发的 bigram 特征 id 列表
    """
    L = len(toks)
    u_feats = []
    for k in range(L):
        ids = []
        for f in expand_features(toks, k, U_tpl):
            j = u2id.get(f, -1)
            if j >= 0:
                ids.append(j)
        u_feats.append(ids)
    b_feats = [[]]  # k=0 没有入边
    for k in range(1, L):
        ids = []
        for f in expand_features(toks, k, B_tpl):
            j = b2id.get(f, -1)
            if j >= 0:
                ids.append(j)
        b_feats.append(ids)
    return u_feats, b_feats


# ---------------- CRF 计分 / 前向-后向 / Viterbi ----------------
def emit_scores(u_feats, W_U):
    """返回 (L, T)，emit[k, t] = sum_{u in u_feats[k]} W_U[u, t]。"""
    L = len(u_feats)
    T = W_U.shape[1]
    em = np.zeros((L, T), dtype=np.float64)
    for k, ids in enumerate(u_feats):
        if ids:
            em[k] = W_U[ids].sum(axis=0)
    return em


def trans_scores(b_feats, W_B):
    """返回 (L, T, T)，trans[k, t1, t2] = sum_{b in b_feats[k]} W_B[b, t1, t2]。
    k=0 全 0（无入边）。
    """
    L = len(b_feats)
    T = W_B.shape[1]
    tr = np.zeros((L, T, T), dtype=np.float64)
    for k in range(1, L):
        ids = b_feats[k]
        if ids:
            tr[k] = W_B[ids].sum(axis=0)
    return tr


def logsumexp_axis(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m.squeeze(axis) + np.log(np.exp(x - m).sum(axis=axis)))


def forward_backward(emit, trans):
    """log-space 前向后向算法。返回 (logZ, log_alpha (L,T), log_beta (L,T))."""
    L, T = emit.shape
    log_alpha = np.empty((L, T), dtype=np.float64)
    log_alpha[0] = emit[0]
    for k in range(1, L):
        # log_alpha[k, t2] = logsumexp_t1 (log_alpha[k-1, t1] + trans[k, t1, t2]) + emit[k, t2]
        scores = log_alpha[k - 1][:, None] + trans[k]  # (T, T)
        log_alpha[k] = logsumexp_axis(scores, axis=0) + emit[k]
    logZ = logsumexp_axis(log_alpha[-1], axis=0)

    log_beta = np.empty((L, T), dtype=np.float64)
    log_beta[-1] = 0.0
    for k in range(L - 2, -1, -1):
        # log_beta[k, t1] = logsumexp_t2 (trans[k+1, t1, t2] + emit[k+1, t2] + log_beta[k+1, t2])
        scores = trans[k + 1] + emit[k + 1][None, :] + log_beta[k + 1][None, :]  # (T, T)
        log_beta[k] = logsumexp_axis(scores, axis=1)
    return float(logZ), log_alpha, log_beta


def viterbi_decode(emit, trans):
    L, T = emit.shape
    delta = np.empty((L, T), dtype=np.float64)
    psi = np.zeros((L, T), dtype=np.int32)
    delta[0] = emit[0]
    for k in range(1, L):
        scores = delta[k - 1][:, None] + trans[k]   # (T_prev, T_cur)
        psi[k] = scores.argmax(axis=0)
        delta[k] = scores.max(axis=0) + emit[k]
    out = [0] * L
    out[-1] = int(delta[-1].argmax())
    for k in range(L - 2, -1, -1):
        out[k] = int(psi[k + 1, out[k + 1]])
    return out


def gold_score(emit, trans, tags):
    s = emit[0, tags[0]]
    for k in range(1, len(tags)):
        s += trans[k, tags[k - 1], tags[k]] + emit[k, tags[k]]
    return s


# ---------------- 训练（SGD on log-linear CRF NLL） ----------------
def sgd_step(u_feats, b_feats, tag_ids, W_U, W_B, lr: float, l2: float):
    """对一句话做一次梯度更新。返回该句的 NLL（用于日志）。
    梯度:
      ∂L/∂W_U[u, t] = ( γ[k, t] - 1[tag_k = t] )  对所有触发 u 的位置 k 求和
      ∂L/∂W_B[b, t1, t2] = ( ξ[k, t1, t2] - 1[tag_{k-1}=t1, tag_k=t2] )  对所有触发 b 的转移位置 k 求和
    L2 正则统一加到所有权重上。
    """
    em = emit_scores(u_feats, W_U)
    tr = trans_scores(b_feats, W_B)
    logZ, log_a, log_b = forward_backward(em, tr)
    nll = logZ - gold_score(em, tr, tag_ids)

    # 边际 γ[k] = exp(log_a[k] + log_b[k] - logZ)
    gamma = np.exp(log_a + log_b - logZ)         # (L, T)
    # 边际 ξ[k]，仅 k>=1: exp(log_a[k-1, t1] + tr[k, t1, t2] + em[k, t2] + log_b[k, t2] - logZ)
    L, T = em.shape

    # 计算每个位置 unigram 梯度对每个 t 的贡献 = gamma - onehot
    delta_em = gamma.copy()
    for k in range(L):
        delta_em[k, tag_ids[k]] -= 1.0  # gamma - empirical

    # 累加 unigram 梯度 → 直接对每个位置触发的 u id 做 in-place 更新
    for k in range(L):
        ids = u_feats[k]
        if not ids:
            continue
        # W_U[ids] -= lr * (delta_em[k] + l2 * W_U[ids])  → 等价于在每行减
        # 对触发的特征施加 L2（仅触发部分；全局 L2 会很慢）
        for j in ids:
            W_U[j] -= lr * (delta_em[k] + l2 * W_U[j])

    # 计算 ξ 并累加 bigram 梯度
    for k in range(1, L):
        ids = b_feats[k]
        if not ids:
            continue
        # ξ[k] = exp(log_a[k-1, :, None] + tr[k] + em[k, None, :] + log_b[k, None, :] - logZ)
        xi = log_a[k - 1][:, None] + tr[k] + em[k][None, :] + log_b[k][None, :] - logZ
        xi = np.exp(xi)                                # (T, T)
        delta_tr = xi.copy()
        delta_tr[tag_ids[k - 1], tag_ids[k]] -= 1.0
        for j in ids:
            W_B[j] -= lr * (delta_tr + l2 * W_B[j])

    return nll


def evaluate(W_U, W_B, sents, U_tpl, B_tpl, u2id, b2id, id2tag, tag2id) -> float:
    from sklearn.metrics import f1_score
    ys, ps = [], []
    for sent in sents:
        toks = [w for w, _ in sent]
        u_feats, b_feats = precompute_sent_features(toks, U_tpl, B_tpl, u2id, b2id)
        em = emit_scores(u_feats, W_U)
        tr = trans_scores(b_feats, W_B)
        pred = viterbi_decode(em, tr)
        for i, t in enumerate(pred):
            ps.append(id2tag[t])
            ys.append(sent[i][1])
    labels = id2tag[1:]  # exclude O
    return float(f1_score(ys, ps, labels=labels, average="micro", zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--min_freq_u", type=int, default=2)
    parser.add_argument("--min_freq_b", type=int, default=5)
    parser.add_argument("--max_train_sents", type=int, default=0,
                        help="0=用全部训练集；调小用于 sanity 调试")
    args = parser.parse_args()

    print("Parsing template:", TEMPLATE_PATH)
    U_tpl, B_tpl = parse_template(TEMPLATE_PATH)
    print(f"  unigram templates={len(U_tpl)}  bigram templates={len(B_tpl)}")

    train_sents = load_corpus(ner_path("Chinese", "train.txt"))
    val_sents = load_corpus(ner_path("Chinese", "validation.txt"))
    if args.max_train_sents > 0:
        train_sents = train_sents[: args.max_train_sents]
    print(f"  train sents={len(train_sents)}  val sents={len(val_sents)}")

    labels = LABELS_CHN
    tag2id = {t: i for i, t in enumerate(labels)}
    T = len(labels)

    print("Building feature index ...")
    t0 = time.time()
    u2id, b2id = build_feature_index(train_sents, U_tpl, B_tpl,
                                     min_freq_u=args.min_freq_u,
                                     min_freq_b=args.min_freq_b)
    F_U, F_B = len(u2id), len(b2id)
    print(f"  |U|={F_U}  |B|={F_B}  (built in {time.time()-t0:.1f}s)")
    print(f"  W_U params={F_U*T}  W_B params={F_B*T*T}  (~{(F_U*T+F_B*T*T)*8/1e6:.1f} MB at fp64)")

    # 预处理所有训练句的 features（占内存但避免重复 hash）
    print("Precomputing per-sentence features ...")
    t0 = time.time()
    train_data = []
    for sent in train_sents:
        toks = [w for w, _ in sent]
        tags = [tag2id[t] for _, t in sent]
        u_feats, b_feats = precompute_sent_features(toks, U_tpl, B_tpl, u2id, b2id)
        train_data.append((u_feats, b_feats, tags))
    print(f"  done in {time.time()-t0:.1f}s")

    W_U = np.zeros((F_U, T), dtype=np.float64)
    W_B = np.zeros((F_B, T, T), dtype=np.float64)

    rng = np.random.RandomState(42)
    best_f1 = -1.0
    t_train = time.time()
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(train_data))
        total_nll = 0.0
        for i, idx in enumerate(order):
            u_feats, b_feats, tags = train_data[idx]
            nll = sgd_step(u_feats, b_feats, tags, W_U, W_B, args.lr, args.l2)
            total_nll += nll
        f1 = evaluate(W_U, W_B, val_sents, U_tpl, B_tpl, u2id, b2id, labels, tag2id)
        elapsed = time.time() - t_train
        print(f"  epoch {ep:02d} | avg-NLL {total_nll/len(train_data):.3f} | "
              f"val micro-F1 {f1:.4f} | elapsed {elapsed:.0f}s")
        if f1 > best_f1:
            best_f1 = f1
            out = os.path.join(HERE, "bonus_template_crf.pkl")
            with open(out, "wb") as f:
                pickle.dump({
                    "W_U": W_U, "W_B": W_B, "u2id": u2id, "b2id": b2id,
                    "tag2id": tag2id, "id2tag": labels,
                    "U_tpl": U_tpl, "B_tpl": B_tpl,
                }, f)
            print(f"    saved best -> {out}")

    print(f"best val micro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
