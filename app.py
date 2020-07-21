import json

import streamlit as st
from pandas import np

from mlas import streamlit_learn

config_path = st.text_input("config file", value="config/rollerball_config.yaml")
run_id = st.text_input("Run ID", value="RollerBall")

if st.button("Train!"):
    options = streamlit_learn.get_run_options(config_path, run_id)
    run_seed = options.env_settings.seed
    if run_seed == -1:
        run_seed = np.random.randint(0, 10000)

    print("Configuration for this run:")
    print(json.dumps(options.as_dict(), indent=4))

    streamlit_learn.run_training(run_seed, options)
