"""Bonus 模板 CRF 推断脚本：加载 pkl，对中文 test.txt 解码并生成预测文件。

输出：bonus/bonus_template_crf_chn_test.txt
"""

from __future__ import annotations

import os
import sys
import pickle
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PJ_ROOT = os.path.dirname(HERE)
if PJ_ROOT not in sys.path:
    sys.path.insert(0, PJ_ROOT)

from data_utils import (  # noqa: E402
    load_tokens_only,
    write_predictions,
    ner_path,
)
from bonus_template_crf import (  # noqa: E402
    precompute_sent_features,
    emit_scores,
    trans_scores,
    viterbi_decode,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="test.txt")
    args = parser.parse_args()

    pkl_path = os.path.join(HERE, "bonus_template_crf.pkl")
    with open(pkl_path, "rb") as f:
        m = pickle.load(f)
    W_U, W_B = m["W_U"], m["W_B"]
    U_tpl, B_tpl = m["U_tpl"], m["B_tpl"]
    u2id, b2id = m["u2id"], m["b2id"]
    id2tag = m["id2tag"]

    src = ner_path("Chinese", args.src)
    sents = load_tokens_only(src)
    preds = []
    t0 = time.time()
    for toks in sents:
        u_feats, b_feats = precompute_sent_features(toks, U_tpl, B_tpl, u2id, b2id)
        em = emit_scores(u_feats, W_U)
        tr = trans_scores(b_feats, W_B)
        pred = viterbi_decode(em, tr)
        preds.append([id2tag[t] for t in pred])
    dt = time.time() - t0

    out = os.path.join(HERE, f"bonus_template_crf_chn_{os.path.splitext(args.src)[0]}.txt")
    write_predictions(src, preds, out)
    print(f"decoded {len(sents)} sents in {dt:.1f}s -> {out}")


if __name__ == "__main__":
    main()
