import json
import queue
import threading

import streamlit as st
from pandas import np
from streamlit.ReportThread import add_report_ctx

from mlas import streamlit_learn

config_path = st.text_input("config file", value="config/rollerball_config.yaml")
run_id = st.text_input("Run ID", value="RollerBall")
force = st.checkbox("Overwrite Existing")

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
    st_command_queue = queue.Queue()
    queue_complete = False
    def st_command_worker():
        while not queue_complete:
            command = st_command_queue.get()
            command()
            st_command_queue.task_done()


    command_thread = threading.Thread(target=st_command_worker)
    add_report_ctx(command_thread)
    command_thread.start()

    streamlit_learn.run_training(run_seed, options, st_command_queue)

    queue_complete = True
    command_thread.join()
