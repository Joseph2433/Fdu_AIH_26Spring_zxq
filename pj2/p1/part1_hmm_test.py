"""Part 1 HMM 推断脚本：加载 pkl，对 test.txt 解码并生成预测文件。

输出：
  p1/part1_hmm_chn_test.txt
  p1/part1_hmm_eng_test.txt

运行：
    cd p1
    python part1_hmm_test.py
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
from part1_hmm import viterbi  # noqa: E402


def predict_file(model: dict, src_path: str, out_path: str):
    sents = load_tokens_only(src_path)
    preds = []
    t0 = time.time()
    for sent in sents:
        preds.append(viterbi(model, sent))
    dt = time.time() - t0
    write_predictions(src_path, preds, out_path)
    return len(sents), sum(len(s) for s in sents), dt


def run(language: str, src_name: str = "test.txt"):
    short = "chn" if language == "Chinese" else "eng"
    pkl = os.path.join(HERE, f"part1_hmm_{short}.pkl")
    with open(pkl, "rb") as f:
        model = pickle.load(f)

    src = ner_path(language, src_name)
    out = os.path.join(HERE, f"part1_hmm_{short}_{os.path.splitext(src_name)[0]}.txt")

    n_sent, n_tok, dt = predict_file(model, src, out)
    print(f"[{language}] decoded {n_sent} sents / {n_tok} tokens in {dt:.1f}s -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="test.txt")
    args = parser.parse_args()
    for lang in ("Chinese", "English"):
        print("=" * 30, lang, "=" * 30)
        run(lang, args.src)


if __name__ == "__main__":
    main()
