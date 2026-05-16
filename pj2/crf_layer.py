"""手写线性链 CRF 层（PyTorch）。Part2 BiLSTM-CRF 与 Part3 BERT-CRF 共用。

接口（与 pytorch-crf 风格一致）：
    crf = LinearChainCRF(num_tags)
    nll = crf.nll_loss(emissions, tags, mask)        # forward 训练损失，标量
    seqs = crf.decode(emissions, mask)               # Viterbi 解码，List[List[int]]
    logZ = crf.partition(emissions, mask)            # log 配分函数（forward 算法）
    score = crf.score(emissions, tags, mask)         # 给定标签序列的非归一 score

形状约定：
    emissions: (B, L, T)   每个位置每个 tag 的发射分数（实数，logit 即可）
    tags:      (B, L) long 真实标签（在 mask 之外的位置可任意）
    mask:      (B, L) bool / 0-1，长度有效位为 1
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """手写线性链 CRF 层。

    Task 2B: 通过 illegal_*_mask 注入 BMES/BIO 硬约束。
        非法位置在 transitions / start / end 上累加 -1e4 大负值（buffer 不可学习），
        保证 forward / score / decode 中这些路径恒被压制；
        loaded checkpoints 自动恢复 buffer 状态。
    """

    def __init__(self, num_tags: int,
                 illegal_trans_mask=None,
                 illegal_start_mask=None,
                 illegal_end_mask=None,
                 neg_inf: float = -1e4):
        super().__init__()
        if num_tags <= 0:
            raise ValueError("num_tags must be positive")
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        # 非法掩码 buffer：非法位置 = neg_inf；合法位置 = 0；与 transitions 等加即可。
        # 用 buffer（非 Parameter）保证不参与 SGD，且 state_dict 自动保存/加载。
        trans_neg = torch.zeros(num_tags, num_tags)
        start_neg = torch.zeros(num_tags)
        end_neg = torch.zeros(num_tags)
        if illegal_trans_mask is not None:
            trans_neg = torch.as_tensor(illegal_trans_mask, dtype=torch.float32) * neg_inf
        if illegal_start_mask is not None:
            start_neg = torch.as_tensor(illegal_start_mask, dtype=torch.float32) * neg_inf
        if illegal_end_mask is not None:
            end_neg = torch.as_tensor(illegal_end_mask, dtype=torch.float32) * neg_inf
        self.register_buffer("trans_neg", trans_neg)
        self.register_buffer("start_neg", start_neg)
        self.register_buffer("end_neg", end_neg)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    # ---- 屏蔽后的有效参数（每次 forward 现算，对训练梯度透明） ----
    def _eff_trans(self):
        return self.transitions + self.trans_neg

    def _eff_start(self):
        return self.start_transitions + self.start_neg

    def _eff_end(self):
        return self.end_transitions + self.end_neg

    # -------- 工具 --------
    @staticmethod
    def _check(emissions: torch.Tensor, tags: Optional[torch.Tensor], mask: Optional[torch.Tensor]):
        assert emissions.dim() == 3
        if tags is not None:
            assert tags.shape == emissions.shape[:2]
        if mask is not None:
            assert mask.shape == emissions.shape[:2]

    def _make_mask(self, emissions: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        return mask.bool()

    # -------- 给定 tag 序列的 unnormalized score --------
    def score(self, emissions: torch.Tensor, tags: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回每个样本的 (start + emission + transition + end) 之和。形状 (B,)。"""
        self._check(emissions, tags, mask)
        mask = self._make_mask(emissions, mask)
        B, L, T = emissions.shape

        eff_trans = self._eff_trans()
        eff_start = self._eff_start()
        eff_end = self._eff_end()

        # 起始
        first = tags[:, 0]                                    # (B,)
        score = eff_start[first]                              # (B,)
        score = score + emissions[torch.arange(B), 0, first]  # 第 0 步发射

        for k in range(1, L):
            prev_tag = tags[:, k - 1]
            cur_tag = tags[:, k]
            trans = eff_trans[prev_tag, cur_tag]
            emit = emissions[torch.arange(B), k, cur_tag]
            step = (trans + emit) * mask[:, k].to(emissions.dtype)
            score = score + step

        # 结束：取每个样本最后一个有效位置的 tag 计入 end_transitions
        last_idx = mask.long().sum(dim=1) - 1                 # (B,)
        last_tag = tags[torch.arange(B), last_idx]            # (B,)
        score = score + eff_end[last_tag]
        return score

    # -------- 配分函数 log Z（forward 算法） --------
    def partition(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回每个样本的 log Z。形状 (B,)。"""
        self._check(emissions, None, mask)
        mask = self._make_mask(emissions, mask)
        B, L, T = emissions.shape

        eff_trans = self._eff_trans()
        eff_start = self._eff_start()
        eff_end = self._eff_end()

        # alpha[b, i] = log P_partial(end=i, obs[:k+1])
        alpha = eff_start.unsqueeze(0) + emissions[:, 0]  # (B, T)
        for k in range(1, L):
            broadcast_alpha = alpha.unsqueeze(2)                 # (B, T, 1)
            broadcast_trans = eff_trans.unsqueeze(0)             # (1, T, T)
            broadcast_emit = emissions[:, k].unsqueeze(1)        # (B, 1, T)
            inner = broadcast_alpha + broadcast_trans + broadcast_emit  # (B, T_prev, T_cur)
            new_alpha = torch.logsumexp(inner, dim=1)            # (B, T_cur)
            m = mask[:, k].unsqueeze(1).to(emissions.dtype)
            alpha = new_alpha * m + alpha * (1 - m)              # 仅在有效位更新

        alpha = alpha + eff_end.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)                     # (B,)

    # -------- 训练损失 --------
    def nll_loss(self, emissions: torch.Tensor, tags: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 reduction: str = "mean") -> torch.Tensor:
        gold = self.score(emissions, tags, mask)
        logZ = self.partition(emissions, mask)
        loss = logZ - gold
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(reduction)

    # -------- Viterbi 解码 --------
    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        self._check(emissions, None, mask)
        mask = self._make_mask(emissions, mask)
        B, L, T = emissions.shape
        device = emissions.device

        eff_trans = self._eff_trans()
        eff_start = self._eff_start()
        eff_end = self._eff_end()

        # delta[b, i] = max log score 到达位置 k, 状态 i
        delta = eff_start.unsqueeze(0) + emissions[:, 0]   # (B, T)
        history: List[torch.Tensor] = []  # 每步的 backpointer

        for k in range(1, L):
            broadcast_delta = delta.unsqueeze(2)                # (B, T_prev, 1)
            broadcast_trans = eff_trans.unsqueeze(0)            # (1, T_prev, T_cur)
            broadcast_emit = emissions[:, k].unsqueeze(1)       # (B, 1, T_cur)
            scores = broadcast_delta + broadcast_trans + broadcast_emit
            best_score, best_prev = scores.max(dim=1)           # (B, T_cur)
            m = mask[:, k].unsqueeze(1).to(emissions.dtype)
            delta = best_score * m + delta * (1 - m)
            history.append(best_prev)

        # +end_transitions
        delta = delta + eff_end.unsqueeze(0)

        # 逐样本回溯（每个样本可能有不同的有效长度）
        seq_lengths = mask.long().sum(dim=1).tolist()
        results: List[List[int]] = []
        for b in range(B):
            Lb = seq_lengths[b]
            if Lb == 0:
                results.append([])
                continue
            # 找最末位置的最佳 tag（不能直接用 delta，因为我们只在有效位更新；
            # 此时 delta[b] 实际就是位置 Lb-1 的 alpha+end）
            last = int(delta[b].argmax().item())
            best_path = [last]
            # 对 b 单样本回溯：遍历 history[k] (k=1..L-1)，但只走有效步
            for k in range(Lb - 1, 0, -1):
                # history 长度是 L-1, history[k-1] 对应从 k-1 → k 的 backpointer
                last = int(history[k - 1][b, last].item())
                best_path.append(last)
            best_path.reverse()
            results.append(best_path)
        return results


# ---------------- 暴力枚举单测 ----------------
def _brute_force(crf: LinearChainCRF, emissions: torch.Tensor,
                 mask: torch.Tensor):
    """对每个样本枚举所有 T^L 条路径，返回 (logZ_brute, best_seq_brute)。仅用于 L 小的测试。"""
    import itertools
    B, L, T = emissions.shape
    Zs, paths = [], []
    seq_lens = mask.long().sum(1).tolist()
    eff_trans = crf._eff_trans().detach()
    eff_start = crf._eff_start().detach()
    eff_end = crf._eff_end().detach()
    for b in range(B):
        Lb = seq_lens[b]
        all_scores = []
        all_seqs = []
        for seq in itertools.product(range(T), repeat=Lb):
            s = float(eff_start[seq[0]] + emissions[b, 0, seq[0]])
            for k in range(1, Lb):
                s += float(eff_trans[seq[k - 1], seq[k]] + emissions[b, k, seq[k]])
            s += float(eff_end[seq[-1]])
            all_scores.append(s)
            all_seqs.append(seq)
        import math
        Z = math.log(sum(math.exp(s) for s in all_scores))
        best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
        Zs.append(Z)
        paths.append(list(all_seqs[best_idx]))
    return Zs, paths


def _self_test():
    torch.manual_seed(0)
    T = 4  # 小 tag 集
    B = 3
    L = 5
    seq_lens = [5, 3, 4]

    # ---- (1) 无约束模式：与暴力枚举对照 ----
    crf = LinearChainCRF(T)
    emissions = torch.randn(B, L, T)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for b, l in enumerate(seq_lens):
        mask[b, :l] = True
    tags = torch.randint(0, T, (B, L))

    logZ_imp = crf.partition(emissions, mask).tolist()
    logZ_bf, paths_bf = _brute_force(crf, emissions, mask)
    print("[no-mask] logZ impl :", [f"{x:.6f}" for x in logZ_imp])
    print("[no-mask] logZ brute:", [f"{x:.6f}" for x in logZ_bf])
    for a, b in zip(logZ_imp, logZ_bf):
        assert abs(a - b) < 1e-4, (a, b)
    paths_imp = crf.decode(emissions, mask)
    print("[no-mask] decode impl :", paths_imp)
    print("[no-mask] decode brute:", paths_bf)
    for pi, pb in zip(paths_imp, paths_bf):
        assert pi == pb

    # 训练性 sanity
    opt = torch.optim.Adam(crf.parameters(), lr=0.1)
    losses = []
    for _ in range(50):
        loss = crf.nll_loss(emissions, tags, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"[no-mask] loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
    assert losses[-1] < losses[0]

    # ---- (2) 带约束模式：禁止 0→0、起始为 1、结束为 2 之外 ----
    illegal_trans = torch.zeros(T, T)
    illegal_trans[0, 0] = 1.0  # 0→0 禁止
    illegal_start = torch.zeros(T)
    illegal_start[2] = 1.0     # 起始不能为 2
    illegal_start[3] = 1.0
    illegal_end = torch.zeros(T)
    illegal_end[0] = 1.0       # 结束不能为 0

    crf2 = LinearChainCRF(T,
                          illegal_trans_mask=illegal_trans,
                          illegal_start_mask=illegal_start,
                          illegal_end_mask=illegal_end)
    # 暴力枚举对照仍应通过（_brute_force 现在用 _eff_*）
    logZ_imp2 = crf2.partition(emissions, mask).tolist()
    logZ_bf2, paths_bf2 = _brute_force(crf2, emissions, mask)
    for a, b in zip(logZ_imp2, logZ_bf2):
        assert abs(a - b) < 1e-3, (a, b)
    paths_imp2 = crf2.decode(emissions, mask)
    for pi, pb in zip(paths_imp2, paths_bf2):
        assert pi == pb
    # 显式验证 Viterbi 输出不违反约束
    for b, p in enumerate(paths_imp2):
        assert illegal_start[p[0]] == 0, f"sample {b} 起始 {p[0]} 非法"
        assert illegal_end[p[-1]] == 0, f"sample {b} 结束 {p[-1]} 非法"
        for k in range(1, len(p)):
            assert illegal_trans[p[k-1], p[k]] == 0, f"sample {b} 转移 {p[k-1]}→{p[k]} 非法"
    print(f"[mask] logZ impl :", [f"{x:.6f}" for x in logZ_imp2])
    print(f"[mask] decode    :", paths_imp2, "(全部合法)")

    # state_dict 保存/加载兼容性
    sd = crf2.state_dict()
    assert "trans_neg" in sd and "start_neg" in sd and "end_neg" in sd
    crf3 = LinearChainCRF(T,
                          illegal_trans_mask=illegal_trans,
                          illegal_start_mask=illegal_start,
                          illegal_end_mask=illegal_end)
    crf3.load_state_dict(sd)
    print("[mask] state_dict round-trip OK, contains buffers:", sorted(k for k in sd if "neg" in k))

    print("LinearChainCRF self-test OK")


if __name__ == "__main__":
    _self_test()
