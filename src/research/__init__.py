"""Rizzmatics research subpackage — scientific validation and hardening.

Everything in here exists to answer one question honestly: *where is Rizzmatics
actually right, where is it wrong, and how much confidence do we deserve?*

Modules:
    experiment  — cross-validated metrics with uncertainty (the workhorse)
    sanity      — normal / shuffled-target / feature-destruction controls
    ablation    — which signal families carry the predictive information
    prefix      — how early does conversational signal become useful
    robustness  — is one variable doing all the work; sensitivity to knobs
    nulldata    — uninformative negative-control dataset
    registry    — reproducible experiment records
    safety      — real-data safety gate
"""
