"""Part 1: 手写 HMM 完成 NER（仅使用 NumPy）。

- 训练：从 NER/{Chinese,English}/train.txt 学 π / A / B（加 α 拉普拉斯平滑）
- 解码：log-space Viterbi
- OOV：发射概率回退到平滑分母（等价 B[t][UNK]）；英文额外做小写 + 形状回退
- 模型文件：part1_hmm_<lang>.pkl

运行：
    cd p1
    python part1_hmm.py
"""

from __future__ import annotations

import os
import sys
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np

# 让 p1/ 内的脚本能 import 到上级 pj2/data_utils.py
HERE = os.path.dirname(os.path.abspath(__file__))
PJ_ROOT = os.path.dirname(HERE)
if PJ_ROOT not in sys.path:
    sys.path.insert(0, PJ_ROOT)

from data_utils import (  # noqa: E402
    LANG2LABELS,
    LANG2SCHEME,
    build_legal_masks,
    load_corpus,
    ner_path,
)


# ---------------- 形状特征（仅英文 OOV 回退使用） ----------------
def english_shape(tok: str) -> str:
    """把英文 token 抽象成一个形状串：A→X, a→x, 0→d, 其他→保留。"""
    out = []
    for ch in tok:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)


def normalize_token(tok: str, language: str) -> str:
    return tok


# ---------------- 训练 ----------------
def train_hmm(language: str, alpha: float = 1e-3,
              unk_threshold: int = 1) -> dict:
    """根据 train.txt 统计 HMM 参数并取 log。

    Task 3 改动：训练时把出现次数 ≤ unk_threshold 的 token 替换为 <UNK>，
    让 <UNK> 学到真实的低频/OOV 发射分布。推断时若 token 不在词表，emission
    回退到 log_B[:, word2id["<UNK>"]]，比原来的均匀 log_oov 更准。

    返回字典含：
      pi[T]       初始分布 log-prob
      A[T,T]      转移 log-prob, A[i,j] = P(t_{k+1}=j | t_k=i)
      B[T,V]      发射 log-prob, B[t,w] = P(w | t)；下标 0 = <UNK>
      log_unk[T]  = log_B[:, 0]，缓存便于推断快速取列
      log_oov[T]  保留：极端兜底（理论上不会再被触发）
      shape_B[T, S] 英文形状回退表（中文为空）
      tag2id, id2tag, word2id, shape2id, language, alpha, unk_threshold
    """
    labels = LANG2LABELS[language]
    tag2id = {t: i for i, t in enumerate(labels)}
    T = len(labels)

    train = load_corpus(ner_path(language, "train.txt"))

    # 第一遍：统计原始 token 频次
    from collections import Counter
    raw_cnt: Counter = Counter()
    for sent in train:
        for tok, _ in sent:
            raw_cnt[tok] += 1

    # Task 3: 把 freq <= unk_threshold 的 token 视为 OOV，统一映射成 <UNK>
    def map_token(tok: str) -> str:
        return tok if raw_cnt[tok] > unk_threshold else "<UNK>"

    # 重新构建词表：<UNK> 占 id 0，其余按频次降序
    word2id: Dict[str, int] = {"<UNK>": 0}
    for tok, c in raw_cnt.most_common():
        if c > unk_threshold:
            word2id[tok] = len(word2id)
    V = len(word2id)

    # 英文形状字典（基于 *原始* token，shape 表保留细粒度）
    shape2id: Dict[str, int] = {"<UNK>": 0}
    if language == "English":
        for tok in raw_cnt:
            sh = english_shape(tok)
            if sh not in shape2id:
                shape2id[sh] = len(shape2id)
    S = len(shape2id)

    # 计数
    init_count = np.zeros(T, dtype=np.float64)
    trans_count = np.zeros((T, T), dtype=np.float64)
    emit_count = np.zeros((T, V), dtype=np.float64)
    shape_count = np.zeros((T, S), dtype=np.float64) if language == "English" else None
    tag_total = np.zeros(T, dtype=np.float64)

    n_unk_replaced = 0
    for sent in train:
        if not sent:
            continue
        first_tag = tag2id[sent[0][1]]
        init_count[first_tag] += 1
        prev_tag = first_tag
        for k, (tok, tag) in enumerate(sent):
            t = tag2id[tag]
            mapped = map_token(tok)
            if mapped == "<UNK>" and tok != "<UNK>":
                n_unk_replaced += 1
            w = word2id.get(mapped, 0)
            emit_count[t, w] += 1
            tag_total[t] += 1
            if shape_count is not None:
                shape_count[t, shape2id[english_shape(tok)]] += 1
            if k > 0:
                trans_count[prev_tag, t] += 1
                prev_tag = t
            else:
                prev_tag = t

    # 加 α 平滑后取 log
    pi = init_count + alpha
    pi /= pi.sum()
    log_pi = np.log(pi)

    A = trans_count + alpha
    A /= A.sum(axis=1, keepdims=True)
    log_A = np.log(A)

    denom = tag_total[:, None] + alpha * V
    B = (emit_count + alpha) / denom
    log_B = np.log(B)
    log_unk = log_B[:, 0]                          # <UNK> 列
    log_oov = np.log(alpha / (tag_total + alpha * V))  # 极端兜底，保留兼容

    if shape_count is not None:
        denom_s = tag_total[:, None] + alpha * S
        shape_B = (shape_count + alpha) / denom_s
        log_shape_B = np.log(shape_B)
    else:
        log_shape_B = None

    # Task 2A: 屏蔽非法 BMES/BIO 转移（HMM 不显式建模 end，所以仅用 legal_init / legal_trans）
    scheme = LANG2SCHEME[language]
    legal_init, legal_trans, _ = build_legal_masks(labels, scheme)
    n_pi_blocked = int((~legal_init).sum())
    n_A_blocked = int((~legal_trans).sum())
    log_pi = np.where(legal_init, log_pi, -np.inf)
    log_A = np.where(legal_trans, log_A, -np.inf)

    print(f"[{language}] UNK threshold={unk_threshold}, "
          f"replaced {n_unk_replaced} train tokens as <UNK> "
          f"(vocab shrank from {len(raw_cnt)+1} → {V}); "
          f"scheme={scheme}, blocked init={n_pi_blocked}/{T}, "
          f"trans={n_A_blocked}/{T*T}")

    return {
        "language": language,
        "alpha": alpha,
        "unk_threshold": unk_threshold,
        "tag2id": tag2id,
        "id2tag": labels,
        "word2id": word2id,
        "shape2id": shape2id,
        "log_pi": log_pi,
        "log_A": log_A,
        "log_B": log_B,
        "log_unk": log_unk,
        "log_oov": log_oov,
        "log_shape_B": log_shape_B,
        "stats": {
            "n_train_sents": len(train),
            "n_train_tokens": int(tag_total.sum()),
            "vocab_size": V,
            "shape_size": S,
            "tag_size": T,
        },
    }


# ---------------- 解码（Viterbi, log-space） ----------------
def viterbi(model: dict, tokens: List[str]) -> List[str]:
    log_pi = model["log_pi"]
    log_A = model["log_A"]
    log_B = model["log_B"]
    log_unk = model.get("log_unk", log_B[:, 0])      # Task 3: <UNK> 学到的发射列
    log_oov = model["log_oov"]
    word2id = model["word2id"]
    shape2id = model["shape2id"]
    log_shape_B = model["log_shape_B"]
    id2tag = model["id2tag"]
    language = model["language"]
    T = log_pi.shape[0]
    L = len(tokens)
    if L == 0:
        return []

    def emission_col(tok: str) -> np.ndarray:
        wid = word2id.get(tok, -1)
        if wid >= 0:
            return log_B[:, wid]
        # OOV
        if language == "English":
            wid_low = word2id.get(tok.lower(), -1)
            if wid_low >= 0:
                return log_B[:, wid_low]
            # 形状回退（cased 信号优先于 <UNK>，因为 shape 含大小写/数字模式）
            sh = english_shape(tok)
            sid = shape2id.get(sh, -1)
            if sid >= 0 and log_shape_B is not None:
                return log_shape_B[:, sid]
        # 最终回退：<UNK> 学到的分布（Task 3，比 log_oov 均匀更准）
        return log_unk

    delta = np.empty((L, T), dtype=np.float64)
    psi = np.zeros((L, T), dtype=np.int32)
    delta[0] = log_pi + emission_col(tokens[0])
    for k in range(1, L):
        # delta[k-1, j] + log_A[j, i] → max over j  →  delta[k, i]
        scores = delta[k - 1][:, None] + log_A  # (T, T) ：行=prev, 列=cur
        psi[k] = scores.argmax(axis=0)
        delta[k] = scores.max(axis=0) + emission_col(tokens[k])

    # 回溯
    out = [0] * L
    out[-1] = int(delta[-1].argmax())
    for k in range(L - 2, -1, -1):
        out[k] = int(psi[k + 1, out[k + 1]])
    return [id2tag[i] for i in out]


# ---------------- 入口 ----------------
def main():
    os.makedirs(HERE, exist_ok=True)
    # Task 3: 中文 freq=1 → <UNK> 训练（+0.31 F1）；
    # 英文很多 freq=1 token 是命名实体本身，1-shot 估计含强信号，关闭替换以保留信号。
    lang_unk = {"Chinese": 1, "English": 0}
    for lang in ("Chinese", "English"):
        t0 = time.time()
        model = train_hmm(lang, alpha=1e-3, unk_threshold=lang_unk[lang])
        dt = time.time() - t0
        out = os.path.join(HERE, f"part1_hmm_{'chn' if lang == 'Chinese' else 'eng'}.pkl")
        with open(out, "wb") as f:
            pickle.dump(model, f)
        s = model["stats"]
        print(
            f"[{lang}] alpha={model['alpha']} unk_th={model['unk_threshold']} | "
            f"sents={s['n_train_sents']} tokens={s['n_train_tokens']} "
            f"V={s['vocab_size']} T={s['tag_size']} shape={s['shape_size']} | "
            f"trained in {dt:.1f}s -> {out}"
        )


if __name__ == "__main__":
    main()
