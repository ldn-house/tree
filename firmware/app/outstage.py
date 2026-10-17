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


def create(n, percent, gamma, order):
    return Output(n, budget_for(n, percent), gamma, order)
