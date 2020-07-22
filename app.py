import json
import queue
import threading
import time

import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.ReportThread import add_report_ctx

from mlas import streamlit_learn
from mlas.stats import StatsEntry


def create_chart():
    df = pd.DataFrame({"step": [0], "mean": [0.0]})
    return alt.Chart(df).mark_line().encode(x="step", y="mean")


"# Unity ML-Agents"

config_path = st.text_input("config file", value="config/rollerball_config.yaml")
run_id = st.text_input("Run ID", value="RollerBall")
force = st.checkbox("Overwrite Existing", value=True)

if st.button("Train!"):
    options = streamlit_learn.get_run_options(config_path, run_id)
    options.checkpoint_settings.force = force

    run_seed = options.env_settings.seed
    if run_seed == -1:
        run_seed = np.random.randint(0, 10000)

    print("Configuration for this run:")
    print(json.dumps(options.as_dict(), indent=4))

    # Create a queue that will process st.writes generated
    # from other threads
    stats_queue = queue.Queue()
    training_complete = False
    training_start_time = time.time()

    chart = st.altair_chart(create_chart())
    def stats_worker():
        while not training_complete:
            entry: StatsEntry = stats_queue.get()
            if "Environment/Cumulative Reward" in entry.values:
                stats_summary = entry.values["Environment/Cumulative Reward"]
                chart.add_rows(pd.DataFrame({"step": [entry.step], "mean": [stats_summary.mean]}))
            stats_queue.task_done()


    command_thread = threading.Thread(target=stats_worker)
    add_report_ctx(command_thread)
    command_thread.start()

    streamlit_learn.run_training(run_seed, options, stats_queue)

    training_complete = True
    command_thread.join()
