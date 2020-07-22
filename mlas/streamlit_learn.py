import os
from queue import Queue
from typing import Dict, Any, Optional

import attr
import mlagents.trainers.cli_utils
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.learn import write_run_options, write_timing_tree, write_training_status, \
    create_environment_factory
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import GaugeWriter, ConsoleWriter, TensorboardWriter, CSVWriter, StatsReporter
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import TrainerFactory, handle_existing_directories
from mlagents.trainers.training_status import GlobalTrainingStatus
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig
from mlagents_envs.timers import hierarchical_timer

from mlas.stats import StreamlitStatsWriter


def run_training(run_seed: int, options: RunOptions, command_queue: Queue) -> None:
    """
    Launches training session.
    :param options: parsed command line arguments
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    with hierarchical_timer("run_training.setup"):
        checkpoint_settings = options.checkpoint_settings
        env_settings = options.env_settings
        engine_settings = options.engine_settings
        base_path = "../results"
        write_path = os.path.join(base_path, checkpoint_settings.run_id)
        maybe_init_path = (
            os.path.join(base_path, checkpoint_settings.initialize_from)
            if checkpoint_settings.initialize_from is not None
            else None
        )
        run_logs_dir = os.path.join(write_path, "run_logs")
        port: Optional[int] = env_settings.base_port
        # Check if directory exists
        handle_existing_directories(
            write_path,
            checkpoint_settings.resume,
            checkpoint_settings.force,
            maybe_init_path,
        )
        # Make run logs directory
        os.makedirs(run_logs_dir, exist_ok=True)
        # Load any needed states
        if checkpoint_settings.resume:
            GlobalTrainingStatus.load_state(
                os.path.join(run_logs_dir, "training_status.json")
            )
        # Configure CSV, Tensorboard Writers and StatsReporter
        # We assume reward and episode length are needed in the CSV.
        csv_writer = CSVWriter(
            write_path,
            required_fields=[
                "Environment/Cumulative Reward",
                "Environment/Episode Length",
            ],
        )
        tb_writer = TensorboardWriter(
            write_path, clear_past_data=not checkpoint_settings.resume
        )
        gauge_write = GaugeWriter()
        console_writer = ConsoleWriter()
        StatsReporter.add_writer(tb_writer)
        StatsReporter.add_writer(csv_writer)
        StatsReporter.add_writer(gauge_write)
        StatsReporter.add_writer(console_writer)
        StatsReporter.add_writer(StreamlitStatsWriter(command_queue))

        if env_settings.env_path is None:
            port = None
        env_factory = create_environment_factory(
            env_settings.env_path,
            engine_settings.no_graphics,
            run_seed,
            port,
            env_settings.env_args,
            os.path.abspath(run_logs_dir),  # Unity environment requires absolute path
        )
        engine_config = EngineConfig(
            width=engine_settings.width,
            height=engine_settings.height,
            quality_level=engine_settings.quality_level,
            time_scale=engine_settings.time_scale,
            target_frame_rate=engine_settings.target_frame_rate,
            capture_frame_rate=engine_settings.capture_frame_rate,
        )
        env_manager = SubprocessEnvManager(
            env_factory, engine_config, env_settings.num_envs
        )
        env_parameter_manager = EnvironmentParameterManager(
            options.environment_parameters, run_seed, restore=checkpoint_settings.resume
        )

        trainer_factory = TrainerFactory(
            options.behaviors,
            write_path,
            not checkpoint_settings.inference,
            checkpoint_settings.resume,
            run_seed,
            env_parameter_manager,
            maybe_init_path,
            False,
        )
        # Create controller and begin training.
        tc = TrainerController(
            trainer_factory,
            write_path,
            checkpoint_settings.run_id,
            env_parameter_manager,
            not checkpoint_settings.inference,
            run_seed,
        )

    # Begin training
    try:
        tc.start_learning(env_manager)
    finally:
        env_manager.close()
        write_run_options(write_path, options)
        write_timing_tree(run_logs_dir)
        write_training_status(run_logs_dir)


def get_run_options(config_path: str, run_id: str) -> RunOptions:
    configured_dict: Dict[str, Any] = {
        "checkpoint_settings": {},
        "env_settings": {},
        "engine_settings": {},
    }

    if config_path is not None:
        config = mlagents.trainers.cli_utils.load_config(config_path)
        configured_dict.update(config)

    # Use the YAML file values for all values not specified in the CLI.
    for key in configured_dict.keys():
        # Detect bad config options
        if key not in attr.fields_dict(RunOptions):
            raise TrainerConfigError(
                "The option {} was specified in your YAML file, but is invalid.".format(
                    key
                )
            )

    configured_dict["checkpoint_settings"]["run_id"] = run_id

    return RunOptions.from_dict(configured_dict)


