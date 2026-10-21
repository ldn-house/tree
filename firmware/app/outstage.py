"""The runtime output stage (docs/effect-api.md, "Output stage").

The implementation is effects/_output.py, shared with the renderer and viewer
so the tree and the previews cap and dim identically. _output_fallback.py is
a byte-identical copy (tests/firmware/test_output_copy.py checks it) used
only if a bundle has no effects/: the power cap must apply whatever else is
broken.
"""

try:
    from effects._output import Output, budget_for, NATIVE, FLOOR_PCT
except Exception:
    from app._output_fallback import Output, budget_for, NATIVE, FLOOR_PCT


def channel_budget(n, limit_a, ma_per_channel, percent=None):
    """effects/_output.Output takes its budget as a sum of channel values
    (0-255 each); the config states the cap in amps. One channel at 255
    draws ma_per_channel mA, so budget = limit_a * 1000 * 255 / ma_per_channel.
    (When effects/_output.py gains a current-based helper, use it here.)"""
    if percent is not None:
        return budget_for(n, percent)
    if not ma_per_channel > 0:
        return 0  # no meaningful current model: dark rather than uncapped
    # Keep fractions (8.9 mA must not become 8) and round down, so the
    # budget never allows more than limit_a.
    return max(0, int(limit_a * 1000.0 * 255 / ma_per_channel))


def estimated_ma(channel_sum, ma_per_channel):
    return channel_sum * ma_per_channel // 255


def estimated_a(channel_sum, ma_per_channel):
    return channel_sum * ma_per_channel / 255000


def create(n, budget, gamma, order, dither="spatial", white=None, floor=FLOOR_PCT):
    """dither: "spatial" (default), "temporal" or False (calibration frames
    pass False). white: optional (r, g, b) balance, each 0..1. floor:
    brightness floor in percent (calibration frames pass 0)."""
    return Output(n, budget, gamma, order, dither, white, floor)
