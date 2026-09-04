# -*- coding: utf-8 -*-
"""CPGAPPO 消融入口别名 — 转发到 train_cpgappo_unified."""
from Algorithms.Train.train_cpgappo_unified import (
    train_cpgappo_dual_cpgappo      as train_cpgappo_main,
    train_cpgappo_dual_wo_guidece   as train_cpgappo_noguidece,
    train_cpgappo_dual_wo_shield    as train_cpgappo_noshield,
    train_cpgappo_dual_wo_appcredit as train_cpgappo_noappcredit,
    train_cpgappo_dual_wo_cpseq     as train_cpgappo_nocp,
    train_cpgappo_dual_forward_only as train_cpgappo_fwdonly,
    train_cpgappo_dual_all_off      as train_cpgappo_alloff,
)

__all__ = [
    "train_cpgappo_main",
    "train_cpgappo_noguidece",
    "train_cpgappo_noshield",
    "train_cpgappo_noappcredit",
    "train_cpgappo_nocp",
    "train_cpgappo_fwdonly",
    "train_cpgappo_alloff",
]
