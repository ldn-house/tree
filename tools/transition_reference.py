"""Plain integer reference for transition kernel parity checks."""

def _mix_py(old, wire, order, n, q, fade, budget):
    i = 0
    total = 0
    while i < n:
        w = q
        if not fade:
            w = (q * 288 // 256 - order[i]) * 8
            if w < 0:
                w = 0
            if w > 256:
                w = 256
        k = i * 3
        end = k + 3
        while k < end:
            v = (old[k] * (256 - w) + wire[k] * w) >> 8
            wire[k] = v
            total += v
            k += 1
        i += 1
    if total > budget:
        k = 0
        while k < n * 3:
            wire[k] = wire[k] * budget // total
            k += 1
