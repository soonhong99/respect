#!/usr/bin/env python3
"""Renderers for paper Figs. 2, 5, and 6 (execution, method overview, and reduction order).

Examples are conceptual manuscript illustrations, not new model outputs.
Execution shares in Fig. 2 use the public summary.
Usage: python scripts/paper_method_figures.py results/paper --out figures
"""
from __future__ import annotations
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EN = True


def save_illustration(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def draw_execution(s, out):
    """Original drawing: plot_execution_261005.py."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        KR = fm.FontProperties(family="DejaVu Sans")
        KRB = fm.FontProperties(family="DejaVu Sans", weight="bold")
        shares = [Decimal(s["sets"][k]["configs"]["S4"]["initial_only_frames"] * 100) / Decimal(s["sets"][k]["n_frames"]) for k in ("event", "general")]
        shares = [v.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) for v in shares]
        matplotlib.use('Agg')
        WORDS = ['Keep', 'distance', 'to', 'the', 'lead', 'vehicle', 'since', 'it', 'is', 'slowing', 'ahead', 'EOS']
        STYLE = {'target': ('#E6E8EB', '#5B6770'), 'prev': ('#C4DFF2', '#1F5F8B'), 'cand': ('#CDEBDD', '#1F7A5A'), 'reject': ('#FFFFFF', '#B0303A')}
        INK, MUTED = ('#1E262C', '#5B6770')
        W, H = (7.1, 3.72)
        CHAR, PAD, GAP, BOX_H = (0.052, 0.1, 0.035, 0.24)
        X0 = 1.42
        BOTTOM = 0.42
        fig = plt.figure(figsize=(W, H - BOTTOM))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(BOTTOM, H)
        ax.axis('off')
        def widths(words):
            return [len(w) * CHAR + PAD for w in words]
        def lefts(words):
            xs, x = ([], X0)
            for w in widths(words):
                xs.append(x)
                x += w + GAP
            return xs
        XS, WS = (lefts(WORDS), widths(WORDS))
        X_END = XS[-1] + WS[-1]
        def box(x, y, w, kind, word, dashed=False, cross=False):
            face, edge = STYLE[kind]
            ax.add_patch(FancyBboxPatch((x, y), w, BOX_H, boxstyle='round,pad=0,rounding_size=0.03', fc=face, ec=edge, lw=0.9, ls=(0, (3, 1.6)) if dashed else '-'))
            ax.text(x + w / 2, y + BOX_H / 2 - 0.005, word, ha='center', va='center', fontsize=7.2, color=INK if not cross else '#8A2A33')
            if cross:
                ax.plot([x + 0.03, x + w - 0.03], [y + 0.04, y + BOX_H - 0.04], color='#B0303A', lw=0.9)
        def row(y, kinds):
            for x, w, word, kind in zip(XS, WS, WORDS, kinds):
                box(x, y, w, kind, word)
        def bracket(i, j, y, text):
            """A bracket under tokens i..j with a label below it."""
            x1, x2 = (XS[i] + 0.01, XS[j] + WS[j] - 0.01)
            ax.plot([x1, x1, x2, x2], [y + 0.05, y, y, y + 0.05], color=MUTED, lw=0.8)
            ax.text((x1 + x2) / 2, y - 0.04, text, ha='center', va='top', fontsize=7.0, fontproperties=KR, color=INK)
        def result(y, main, sub=None):
            ax.text(X_END + 0.16, y + BOX_H / 2 + (0.07 if sub else 0), main, ha='left', va='center', fontsize=8.2, fontproperties=KRB, color=INK)
            if sub:
                ax.text(X_END + 0.16, y + BOX_H / 2 - 0.12, sub, ha='left', va='center', fontsize=7.0, fontproperties=KR, color=MUTED)
        def case_label(y, title, sub, share=None):
            dy = 0.08 if share else 0.0
            ax.text(0.12, y + BOX_H / 2 + 0.07 + dy, title, ha='left', va='center', fontsize=7.8, fontproperties=KRB, color=INK)
            if share:
                ax.text(0.12, y + BOX_H / 2 - 0.09 + dy, share, ha='left', va='center', fontsize=7.2, fontproperties=KRB, color=INK)
            ax.text(0.12, y + BOX_H / 2 - 0.1 - (0.9 * dy if share else 0), sub, ha='left', va='center', fontsize=6.2, fontproperties=KR, color=MUTED)
        def legend(y, items):
            x = X0
            for kind, dashed, label in items:
                face, edge = STYLE[kind]
                ax.add_patch(FancyBboxPatch((x, y - 0.055), 0.15, 0.11, boxstyle='round,pad=0,rounding_size=0.02', fc=face, ec=edge, lw=0.8, ls=(0, (3, 1.6)) if dashed else '-'))
                t = ax.text(x + 0.21, y, label, ha='left', va='center', fontsize=7.0, fontproperties=KR, color=INK)
                fig.canvas.draw()
                bb = t.get_window_extent().transformed(ax.transData.inverted())
                x = bb.x1 + 0.28
        YA = 3.12
        legend(H - 0.12, [('target', False, 'token computed by the target model')])
        case_label(YA, 'Every case', 'one target run per token')
        row(YA, ['target'] * len(WORDS))
        for k, (x, w) in enumerate(zip(XS, WS), start=1):
            ax.text(x + w / 2, YA - 0.05, str(k), ha='center', va='top', fontsize=6.4, color=MUTED)
        result(YA, '12 target runs')
        ax.text(W / 2, YA - 0.34, '(a) Autoregressive decoding', ha='center', va='center', fontsize=8.6, fontproperties=KR, color=INK)
        ax.plot([0.12, W - 0.12], [YA - 0.5, YA - 0.5], color='#D5DADE', lw=0.7)
        legend(YA - 0.67, [('prev', False, 'accepted from a previous CoC'), ('cand', False, 'accepted from draft-model or datastore candidates'), ('reject', True, 'rejected candidate')])
        def previous(y, sentence):
            ax.text(X0, y, 'Previous CoC:  ' + sentence, ha='left', va='bottom', fontsize=6.8, fontproperties=KR, color=MUTED)
        YB1 = 1.93
        case_label(YB1, 'Common case', 'previous CoC fully matches', f'({shares[0]}%, {shares[1]}%)')
        previous(YB1 + BOX_H + 0.05, 'Keep distance to the lead vehicle since it is slowing ahead')
        row(YB1, ['prev'] * len(WORDS))
        bracket(0, len(WORDS) - 1, YB1 - 0.06, '① Initial verification: the whole previous CoC in one target run')
        result(YB1, '1 target run', '1 initial verification')
        YB2 = 0.97
        case_label(YB2, 'Continuation needed', 'previous CoC diverges', f'({100 - shares[0]}%, {100 - shares[1]}%)')
        previous(YB2 + BOX_H + 0.34, 'Keep distance to the lead vehicle since it is directly ahead in our lane')
        row(YB2, ['prev'] * 9 + ['target', 'cand', 'cand'])
        rej_w = widths(['directly'])[0]
        box(XS[9] + WS[9] / 2 - rej_w / 2, YB2 + BOX_H + 0.05, rej_w, 'reject', 'directly', dashed=True, cross=True)
        bracket(0, 9, YB2 - 0.06, '① Initial verification')
        bracket(10, 11, YB2 - 0.06, '② Continuation × k')
        result(YB2, '1 + k target runs', '1 initial + k continuation verifications')
        ax.text(W / 2, 0.55, '(b) RESPECT Decode', ha='center', va='center', fontsize=8.6, fontproperties=KR, color=INK)
        save_illustration(fig, out, "fig2_execution")


def draw_architecture(s, out):
    """Original drawing: plot_architecture.py."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        KR = fm.FontProperties(family="DejaVu Sans")
        KRB = fm.FontProperties(family="DejaVu Sans", weight="bold")
        matplotlib.use('Agg')
        plt.rcParams['hatch.linewidth'] = 0.45
        STYLE = {'prefill': ('#FFFFFF', '#8A949B', '#1E262C'), 'target': ('#E6E8EB', '#5B6770', '#1E262C'), 'prev': ('#C4DFF2', '#1F5F8B', '#1E262C'), 'cand': ('#CDEBDD', '#1F7A5A', '#1E262C'), 'reject': ('#FFFFFF', '#B0303A', '#8A2A33'), 'skip': ('#FFFFFF', '#AEB6BC', '#9AA3AA'), 'plain': ('#FFFFFF', '#5B6770', '#1E262C')}
        INK, MUTED, LINE = ('#1E262C', '#5B6770', '#D5DADE')
        KERNEL, FDC, DSC = ('#176C4E', '#6B55A0', '#1F7A5A')
        W, H = (7.1, 6.08)
        CHAR, PAD, GAP, BH = (0.047, 0.085, 0.028, 0.2)
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis('off')
        def kr(x, y, s, size=6.8, bold=False, color=INK, **kw):
            kw.setdefault('ha', 'left')
            kw.setdefault('va', 'center')
            return ax.text(x, y, s, fontsize=size, fontproperties=KRB if bold else KR, color=color, **kw)
        def right_of(t, pad=0.0):
            fig.canvas.draw()
            return t.get_window_extent().transformed(ax.transData.inverted()).x1 + pad
        def wlen(word):
            return len(word) * CHAR + PAD
        def text_w(word, size=6.4):
            t = ax.text(0, 0, word, fontsize=size)
            fig.canvas.draw()
            w = t.get_window_extent().transformed(ax.transData.inverted()).width
            t.remove()
            return w
        def box(x, y, word, kind, size=6.4):
            face, edge, ink = STYLE[kind]
            w = max(wlen(word), text_w(word, size) + 0.05) if EN else wlen(word)
            ax.add_patch(FancyBboxPatch((x, y), w, BH, boxstyle='round,pad=0,rounding_size=0.025', fc=face, ec=edge, lw=0.8, ls=(0, (3, 1.5)) if kind in ('reject', 'skip') else '-', hatch='//////' if kind == 'prefill' else None))
            bb = dict(boxstyle='square,pad=0.05', fc='#FFFFFF', ec='none') if kind == 'prefill' else None
            ax.text(x + w / 2, y + BH / 2 - 0.004, word, ha='center', va='center', fontsize=size, color=ink, bbox=bb)
            if kind == 'reject':
                ax.plot([x + 0.025, x + w - 0.025], [y + 0.035, y + BH - 0.035], color=edge, lw=0.8)
            return x + w
        def row(x, y, words, kinds):
            lefts = []
            for word, kind in zip(words, kinds):
                lefts.append(x)
                x = box(x, y, word, kind) + GAP
            return (lefts, x - GAP)
        def arrow(p, q, color=MUTED, lw=0.8, ls='-', rad=0.0, head=6):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=head, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))
        def panel(x, y, w, h, edge=LINE, face='none', lw=0.8, ls='-'):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=face, ec=edge, lw=lw, ls=ls))
        def outline(x0, x1, y, color, pad):
            """Dashed outline around boxes that are reused by a later step."""
            ax.add_patch(FancyBboxPatch((x0 - pad, y - pad), x1 - x0 + 2 * pad, BH + 2 * pad, boxstyle='round,pad=0,rounding_size=0.04', fc='none', ec=color, lw=0.9, ls=(0, (2.5, 1.5)), zorder=5))
        def section(y, title, note, right=W - 0.1):
            t = kr(0.1, y, title, size=8.4, bold=True)
            kr(right_of(t, 0.14), y, note, size=6.9, color=MUTED)
            ax.plot([0.1, right], [y - 0.14, y - 0.14], color=LINE, lw=0.7)
        def badge(x, y, n):
            ax.add_patch(plt.Circle((x, y), 0.072, fc='#2F3A42', ec='none', zorder=6))
            ax.text(x, y - 0.004, str(n), ha='center', va='center', fontsize=6.2 if len(str(n)) == 1 else 5.4, color='#FFFFFF', fontweight='bold', zorder=7)
        def step_title(x, y, n, title, note=None, size=7.0):
            """Every reading step uses the same mark: badge, bold title, optional muted note."""
            badge(x, y, n)
            t = kr(x + 0.13, y, title, size=size, bold=True)
            if note:
                kr(right_of(t, 0.1), y, note, size=6.2, color=MUTED)
        def step_ref(x, y, before, n, after, size=5.9, color=MUTED):
            t = kr(x, y, before, size=size, color=color)
            x = right_of(t, 0.09) if before else x + 0.07
            badge(x, y, n)
            kr(x + 0.1, y, after, size=size, color=color)
        def tokens_after(x, y, label, words, kinds, size=6.3):
            t = kr(x, y + BH / 2, label, size=size)
            return row(right_of(t, 0.06), y, words, kinds)
        def legend(y, items):
            x = 0.1
            for kind, label in items:
                face, edge, _ = STYLE[kind]
                ax.add_patch(FancyBboxPatch((x, y - 0.05), 0.14, 0.1, boxstyle='round,pad=0,rounding_size=0.02', fc=face, ec=edge, lw=0.8, ls=(0, (3, 1.5)) if kind in ('reject', 'skip') else '-', hatch='//////' if kind == 'prefill' else None))
                x = right_of(kr(x + 0.19, y, label, size=6.7), 0.22)
        def branch(x0, y0, x1, ys, color=MUTED):
            for yy in ys:
                ax.plot([x0, x1], [y0 + BH / 2, yy + BH / 2], color=color, lw=0.8)
        legend(H - 0.13, [('prefill', 'first token determined by Prefill'), ('target', 'correction token (computed by the target model)'), ('prev', 'accepted from a previous CoC')])
        legend(H - 0.33, [('cand', 'accepted from FD-DFlash or datastore candidates'), ('reject', 'rejected candidate'), ('plain', 'candidate before verification')])
        kr(0.1, H - 0.58, 'Example:', size=7.2, bold=True)
        kr(right_of(ax.texts[-1], 0.14) if EN else 1.3, H - 0.58, 'CoC to be generated at frame $t$:  ' + 'Accelerate to proceed through the intersection since the straight traffic light is green', size=7.0)
        KX0 = 4.9 if EN else 5.02
        Y1 = H - 0.92
        Y2 = Y1 - 2.02
        section(Y1, '① Initial verification', 'candidates from the previous-$N$-frame CoCs', right=KX0 - 0.12)
        TEXT = {'A': 'Keep distance to the lead vehicle …', 'B': 'Accelerate to proceed through …', 'C': 'Accelerate and keep pace with …'}
        TAGC = {'A': '#F1F3F5', 'B': '#FBEEDC', 'C': '#E3F0E8'}
        history = ['A', 'A', 'C', 'A', 'A', 'B', 'A', 'B']
        step_title(0.17, Y1 - 0.33, 1, 'Previous CoCs')
        kr(0.30, Y1 - 0.44, 'previous $N$ frames (e.g., $N$ = 4)', size=5.2, color=MUTED)
        def tag(x, y, t):
            ax.add_patch(FancyBboxPatch((x, y - 0.055), 0.13, 0.11, boxstyle='round,pad=0,rounding_size=0.02', fc=TAGC[t], ec='#8A949B', lw=0.6))
            ax.text(x + 0.065, y - 0.003, t, ha='center', va='center', fontsize=6.0, color=INK)
        for k, t in enumerate(history):
            y = Y1 - 0.53 - k * 0.155
            kr(0.12, y, f'$t{k - 8}$', size=6.2, color=MUTED)
            tag(0.42, y, t)
            ax.text(0.62, y, TEXT[t], ha='left', va='center', fontsize=6.0, color=MUTED)
        step_title(2.37, Y1 - 0.33, 2, 'Initial trie')
        TX, ry = (2.3, Y1 - 1.05)
        yB, yC = (ry + 0.26, ry - 0.26)
        arrow((2.17 if EN else 2.12, ry + BH / 2), (TX - 0.03, ry + BH / 2))
        r_end = box(TX, ry, 'Accelerate', 'prefill')
        nb = r_end + 0.12
        branch(r_end, ry, nb, (yB, yC))
        _, b_end = row(nb, yB, ['to', '…', 'light', 'turns', '…', 'EOS'], ['plain'] * 6)
        _, c_end = row(nb, yC, ['and', 'keep', '…', 'EOS'], ['plain'] * 4)
        section(Y2, '② Continuation verification', 'candidates from FD-DFlash and the datastore', right=KX0 - 0.12)
        SX, SW, S_top = (0.1, 1.5, Y2 - 0.95)
        panel(SX, S_top - 0.66, SW, 0.66, face='#F7F8F9')
        step_title(SX + 0.15, S_top - 0.16, 4, 'Accepted state')
        s_row = S_top - 0.52
        row(SX + 0.08, s_row, ['Accelerate', '…', 'light', 'is'], ['prefill', 'prev', 'prev', 'target'])
        PX, PW = (1.78, 1.58) if EN else (1.86, 1.52)
        F_top = Y2 - 0.24
        panel(PX, F_top - 1.24, PW, 1.24, edge=FDC)
        step_title(PX + 0.15, F_top - 0.14, 5, 'FD-DFlash prediction')
        if EN:
            kr(PX + 0.08, F_top - 0.36, 'Input: correction token', size=6.0)
            t = kr(PX + 0.08, F_top - 0.52, '+ features of', size=6.0)
            badge(right_of(t, 0.09), F_top - 0.52, 4)
            kr(PX + 0.08, F_top - 0.7, 'Prediction:', size=6.3)
            fx = PX + 0.1
        else:
            t = kr(PX + 0.08, F_top - 0.4, '입력: 정정 token +', size=6.0)
            badge(right_of(t, 0.09), F_top - 0.4, 4)
            kr(right_of(t, 0.19), F_top - 0.4, '의 target 특징', size=6.0)
            t = kr(PX + 0.08, F_top - 1.01 + BH / 2, '예측:', size=6.3)
            fx = right_of(t, 0.05)
        t_end = box(fx, F_top - 1.01, 'turning', 'plain')
        branch(t_end, F_top - 1.01, t_end + 0.08, (F_top - 0.88, F_top - 1.14))
        row(t_end + 0.08, F_top - 0.88, ['green', '…'], ['plain'] * 2)
        row(t_end + 0.08, F_top - 1.14, ['yellow'], ['plain'])
        arrow((SX + SW, S_top - 0.25), (PX - 0.01, F_top - 0.62))
        D_top = F_top - 1.34
        panel(PX, D_top - 0.74, PW, 0.74, edge=DSC)
        step_title(PX + 0.15, D_top - 0.14, '5′', 'Datastore lookup')
        kr(PX + 0.08, D_top - 0.34, 'key: last accepted + correction token', size=6.0)
        tokens_after(PX + 0.08, D_top - 0.66, 'after the key:', ['green', 'EOS'], ['plain'] * 2)
        arrow((SX + SW, S_top - 0.5), (PX - 0.01, D_top - 0.37))
        MX = 3.5 if EN else 3.66
        m_root = Y2 - 1.4
        step_title(MX + 0.05, Y2 - 0.4, 6, 'Continuation trie')
        m_end = box(MX, m_root, 'is', 'target')
        mb = m_end + 0.12
        y_ds, y_fd = (m_root + 0.42, m_root - 0.3)
        branch(m_end, m_root, mb, (y_ds,), color=DSC)
        branch(m_end, m_root, mb, (y_fd,), color=FDC)
        _, ds_end = row(mb, y_ds, ['green', 'EOS'], ['plain'] * 2)
        tf_end = box(mb, y_fd, 'turning', 'plain')
        branch(tf_end, y_fd, tf_end + 0.08, (y_fd + 0.14, y_fd - 0.14), color=FDC)
        _, fd_end = row(tf_end + 0.08, y_fd + 0.14, ['green', '…'], ['plain'] * 2)
        row(tf_end + 0.08, y_fd - 0.14, ['yellow'], ['plain'])
        arrow((PX + PW, F_top - 0.62), (MX - 0.02, m_root + BH / 2 + 0.05), color=FDC, rad=-0.15)
        arrow((PX + PW, D_top - 0.4), (MX - 0.02, m_root + BH / 2 - 0.05), color=DSC, rad=0.15)
        KW = W - 0.1 - KX0
        H1, H2 = (1.14, 0.9)
        c1, c2 = (Y1 - 0.34, Y2 - 0.62)
        col_top = Y1 + 0.12
        col_bot = min(D_top - 0.74, c2 - H2 - 0.5) - 0.08
        panel(KX0, col_bot, KW, col_top - col_bot, edge=KERNEL, face='#F2F8F5', lw=1.0)
        kr(KX0 + KW / 2, Y1, 'Target run on deterministic kernels', size=7.2, bold=True, color=KERNEL, ha='center')
        kr(KX0 + KW / 2, Y1 - 0.19, 'the whole trie verified in one target run', size=6.3, color=KERNEL, ha='center')
        CX, CWD = (KX0 + 0.1, KW - 0.2)
        def card(top, height, title, step):
            panel(CX, top - height, CWD, height, edge='#9CCBB5', face='#FFFFFF')
            step_title(CX + 0.17, top - 0.18, step, title)
        def decision(bottom):
            """Diamond 'EOS 수용?' centred under a card; 'no' exits left to (4), 'yes' exits right to (8)."""
            dx, dy = (CX + CWD / 2, bottom - 0.3)
            arrow((dx, bottom - 0.01), (dx, dy + 0.15), color=INK, lw=0.8, head=5)
            ax.add_patch(plt.Polygon([(dx - 0.24, dy), (dx, dy + 0.15), (dx + 0.24, dy), (dx, dy - 0.15)], closed=True, fc='#FFFFFF', ec=INK, lw=0.8, zorder=4))
            ax.text(dx, dy - 0.004, 'EOS?', ha='center', va='center', fontsize=5.4, fontproperties=KR, linespacing=0.95, zorder=5)
            for side, label, n in ((-1, 'no', 4), (1, 'yes', 8)):
                x0, x1 = (dx + side * 0.24, dx + side * 0.62)
                arrow((x0, dy), (x1 - side * 0.08, dy), color=INK, lw=0.8, head=5)
                kr((x0 + x1) / 2, dy + 0.08, label, size=6.0, ha='center')
                badge(x1, dy, n)
        card(c1, H1, 'Target run (initial)', 3)
        tokens_after(CX + 0.1, c1 - 0.52, 'Accepted:', ['Accelerate', 'to', '…', 'light'], ['prefill', 'prev', 'prev', 'prev'])
        tokens_after(CX + 0.1, c1 - 0.78, 'Rejected:', ['and', 'turns'], ['reject'] * 2)
        tokens_after(CX + 0.1, c1 - 1.04, 'Correction token:', ['is'], ['target'])
        decision(c1 - H1)
        arrow((max(b_end, c_end) + 0.28, ry + BH / 2), (CX - 0.01, ry + BH / 2), color=KERNEL)
        card(c2, H2, 'Target run (continuation)', 7)
        tokens_after(CX + 0.1, c2 - 0.52, 'Accepted:', ['green', 'EOS'], ['cand'] * 2)
        tokens_after(CX + 0.1, c2 - 0.78, 'Rejected:', ['turning'], ['reject'])
        decision(c2 - H2)
        arrow((max(ds_end, fd_end) + 0.05, m_root + BH / 2), (CX - 0.01, m_root + BH / 2), color=KERNEL)
        yl = col_bot
        YF = yl - 0.42
        ax.plot([0.1, W - 0.1], [YF + BH + 0.14, YF + BH + 0.14], color=LINE, lw=0.7)
        step_title(0.17, YF + BH / 2, 8, 'Final CoC')
        words = ['Accelerate', 'to', 'proceed', 'through', 'the', 'intersection', 'since', 'the', 'straight', 'traffic', 'light', 'is', 'green', 'EOS']
        _, f_end = row(0.92, YF, words, ['prefill'] + ['prev'] * 10 + ['target', 'cand', 'cand'])
        save_illustration(fig, out, "fig5_method")


def draw_kernel_order(s, out):
    """Original drawing: plot_kernel_order.py."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        KR = fm.FontProperties(family="DejaVu Sans")
        KRB = fm.FontProperties(family="DejaVu Sans", weight="bold")
        matplotlib.use('Agg')
        PAIR_COLORS = ('#426685', '#8964ae')
        LOGICAL = '#176C4E'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42})
        titles = ['(a) Adjacent pairing,\ncontiguous', '(b) Adjacent pairing,\nscattered', '(c) Logical-order accumulation,\ncontiguous', '(d) Logical-order accumulation,\nscattered']
        fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.55))
        for i, ax in enumerate(axes.flat):
            ax.set(xlim=(-0.5, 7.5), ylim=(0, 1.04))
            ax.axis('off')
            logical, scattered = (i >= 2, i % 2)
            color = LOGICAL if logical else PAIR_COLORS[scattered]
            values = ['a', 'b', 'c', '·', '·', '·', '·', 'd'] if scattered else ['a', 'b', 'c', 'd', '·', '·', '·', '·']
            ax.text(3.5, -0.1, titles[i], ha='center', va='top', fontproperties=KR, linespacing=1.15).set_fontsize(6.2)
            for x, v in enumerate(values):
                ax.text(x, 0.8, v, ha='center', va='center', fontsize=6.6, bbox=dict(boxstyle='square,pad=.18', fc='white', ec=color if v != '·' else '#ccd0d5'))
                ax.text(x, 0.99, str(x), ha='center', fontsize=5, color='#888')
            ax.text(-0.5, 0.99, 'Slot', ha='right', color='#888', fontproperties=KR).set_fontsize(5)
            if logical:
                ax.plot([0, 7], [0.51, 0.51], c=color, lw=1.5)
                for x, v in enumerate(values):
                    if v != '·':
                        ax.plot([x, x], [0.69, 0.51], c=color)
                        ax.plot(x, 0.51, 'o', c=color, ms=2.6)
                expr = '(((a + b) + c) + d)'
            else:
                for x in (0, 2, 4, 6):
                    ax.plot([x, x + 0.5, x + 1], [0.69, 0.51, 0.69], c=color, lw=1)
                ax.plot([0.5, 1.5, 2.5], [0.51, 0.37, 0.51], c=color)
                ax.plot([4.5, 5.5, 6.5], [0.51, 0.37, 0.51], c=color)
                ax.plot([1.5, 3.5, 5.5], [0.37, 0.24, 0.37], c=color)
                expr = '(((a + b) + c) + d)' if scattered else '((a + b) + (c + d))'
            ax.text(3.5, 0.04, expr, ha='center', fontsize=7, color=color)
        fig.tight_layout(h_pad=2.6 if EN else 1.6, w_pad=0.8)
        fig.subplots_adjust(bottom=0.15 if EN else 0.1)
        save_illustration(fig, out, "fig6_kernel_order")

def draw_method_figures(summary, out):
    out.mkdir(parents=True, exist_ok=True)
    draw_execution(summary, out)
    draw_architecture(summary, out)
    draw_kernel_order(summary, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    summary = json.loads((Path(args.root) / "summary.json").read_text())
    draw_method_figures(summary, Path(args.out))


if __name__ == "__main__":
    main()
