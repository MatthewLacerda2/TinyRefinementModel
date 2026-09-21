"""The verifiable worlds the model will learn in by trying (#440).

A world is three things and no more: tasks it can be given, a verifier that says
whether an attempt worked, and a readout that turns many attempts into a number.
Nothing here trains anything, and nothing here asks a model to judge a model —
"verifiable" means a test either passed or it did not.

Python is the first world. A formal-proof language is meant to be the second, in
the same shape.
"""
