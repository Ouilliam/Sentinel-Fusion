"""
    ./utils.py

        Contains definitions of utils functions such as a 
        progress bar 
"""
import os
import random

import numpy as np
import torch


def get_progress_bar(idx, n_batch, bar_size: int = 20):
    """
        Function that returns a progress bar of the current 
        epoch.

        Parameters:
        -----------
        idx: int
            The current index of the epoch
        batch_size: int
            The batch size of the current epoch
        bar_size: int
            The size of the progress bar

        Returns:
        --------
        str
            The progress bar
    """
    filled = int((idx / n_batch) * bar_size)
    bar = ("=" * filled + ">") + " " * (bar_size - filled)
    return f"[{bar}] {idx}/{n_batch}"
