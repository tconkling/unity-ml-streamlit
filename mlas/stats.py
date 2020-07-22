import time
from queue import Queue
from typing import Dict, Any

import streamlit as st
from mlagents.trainers.stats import StatsWriter, StatsSummary, StatsPropertyType


class StreamlitStatsWriter(StatsWriter):
    def __init__(self, command_queue: Queue):
        self.training_start_time = time.time()
        # If self-play, we want to print ELO as well as reward
        self.self_play = False
        self.self_play_team = -1
        self.command_queue = command_queue

    def write(self, *args):
        self.command_queue.put(lambda: st.write(*args))

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        is_training = "Not Training."
        if "Is Training" in values:
            stats_summary = values["Is Training"]
            if stats_summary.mean > 0.0:
                is_training = "Training."

        if "Environment/Cumulative Reward" in values:
            stats_summary = values["Environment/Cumulative Reward"]
            time_elapsed = time.time() - self.training_start_time
            self.write(
                f"{category}: Step: {step}. "
                f"Time Elapsed: {time_elapsed:0.3f} s "
                f"Mean Reward: {stats_summary.mean:0.3f}"
                f". Std of Reward: {stats_summary.std:0.3f}. {is_training}"
            )
            if self.self_play and "Self-play/ELO" in values:
                elo_stats = values["Self-play/ELO"]
                self.write(f"{category} ELO: {elo_stats.mean:0.3f}. ")
        else:
            self.write(
               f"{category}: Step: {step}. No episode was completed since last summary. {is_training}"
            )

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            self.write(f"Hyperparameters for behavior name {category}:")
            self.write(value)
        elif property_type == StatsPropertyType.SELF_PLAY:
            assert isinstance(value, bool)
            self.self_play = value
