"""The runtime output stage (docs/effect-api.md, "Output stage").

The implementation is effects/_output.py, shared with the renderer and viewer
so the tree and the previews cap and dim identically. _output_fallback.py is
a byte-identical copy (tests/firmware/test_output_copy.py checks it) used
only if a bundle has no effects/: the power cap must apply whatever else is
broken.
"""

try:
    from effects._output import Output, budget_for
except Exception:
    from app._output_fallback import Output, budget_for


def channel_budget(n, budget_ma, ma_per_channel, percent=None):
    """The output stage's budget is a sum of channel values (0-255 each);
    the config states it as current. One channel at 255 draws
    ma_per_channel, so budget = budget_ma * 255 / ma_per_channel."""
    if percent is not None:
        return budget_for(n, percent)
    return max(0, int(budget_ma * 255 // max(1, ma_per_channel)))


def estimated_ma(channel_sum, ma_per_channel):
    return channel_sum * ma_per_channel // 255


def create(n, budget, gamma, order):
    return Output(n, budget, gamma, order)
