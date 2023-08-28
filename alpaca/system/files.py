from typing import Union, Generator
import os
from os.path import join, basename, dirname, abspath, splitext, isfile, relpath
import glob
import string
import re

from shutil import rmtree

from alpaca.system.utils.filesystem import djoin
from alpaca.system._session import ActiveLearningSession
from alpaca.system.factory import (
    DataState,
    ModelState,
    SelectorState,
    ExperimentState,
)

from alpaca.system.reports import IterationReport

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
DATA_DIR = djoin(BASE_DIR, "data")
MODELS_DIR = djoin(BASE_DIR, "models")
SESSIONS_DIR = djoin(BASE_DIR, "sessions")
DATA_STATES_DIR = djoin(SESSIONS_DIR, ".data_states")
MODEL_STATES_DIR = djoin(SESSIONS_DIR, ".model_states")
SELECTOR_STATES_DIR = djoin(SESSIONS_DIR, ".selector_states")


def _dir_is_empty(dirname: str) -> bool:
    return len(glob.glob(dirname)) <= 1


VALID_NAME_CHARS = set(string.ascii_letters + string.digits + "." + "_")


def _check_name_contains_valid_chars(name: str) -> str:
    return all(c in VALID_NAME_CHARS for c in name)


def _get_session_dir(session_name: str) -> str:
    subdirs = session_name.split(".")
    return djoin(SESSIONS_DIR, *subdirs)


def _get_iteration_report_path(session_name: str, iteration: Union[int, str]) -> str:
    return join(
        _get_session_dir(session_name),
        f"iteration_report_{iteration}.json",
    )


def _get_experiment_state_path(session_name: str, iteration: Union[int, str]) -> str:
    return join(
        _get_session_dir(session_name),
        f"experiment_state_{iteration}.json",
    )


def _get_saved_model_params_path(session_name: str, iteration: int) -> str:
    return join(_get_session_dir(session_name), f"model_{iteration}.params")


def _get_saved_model_state_path(session_name: str, iteration: int) -> str:
    return join(_get_session_dir(session_name), f"model_{iteration}.state")


class SessionException(Exception):
    ...


def _delete(path) -> None:
    try:
        rmtree(path)  # No error if path is a directory
    except NotADirectoryError:  # Path is a file
        os.remove(path)
    except FileNotFoundError:  # Path does not exist
        pass


class FileUtils:
    def __init__(self):
        raise RuntimeError("Storage is a namespace class and cannot be instantiated")

    @staticmethod
    def save_data_state(data_state: DataState, name: str) -> None:
        _check_name_contains_valid_chars(name)
        data_state.save(join(DATA_STATES_DIR, f"{name}.json"))

    @staticmethod
    def load_data_state(name: str) -> DataState:
        return DataState.load(join(DATA_STATES_DIR, f"{name}.json"))

    @staticmethod
    def delete_data_state(name: str) -> None:
        _delete(join(DATA_STATES_DIR, f"{name}.json"))

    @staticmethod
    def save_model_state(model_state: ModelState, name: str) -> None:
        _check_name_contains_valid_chars(name)
        model_state.save(join(MODEL_STATES_DIR, f"{name}.json"))

    @staticmethod
    def load_model_state(name: str) -> ModelState:
        return ModelState.load(join(MODEL_STATES_DIR, f"{name}.json"))

    @staticmethod
    def delete_model_state(name: str) -> None:
        _delete(join(MODEL_STATES_DIR, f"{name}.json"))

    @staticmethod
    def save_selector_state(selector_state: SelectorState, name: str) -> None:
        _check_name_contains_valid_chars(name)
        selector_state.save(join(SELECTOR_STATES_DIR, f"{name}.json"))

    @staticmethod
    def load_selector_state(name: str) -> SelectorState:
        return SelectorState.load(join(SELECTOR_STATES_DIR, f"{name}.json"))

    @staticmethod
    def delete_selector_state(name: str) -> None:
        _delete(join(SELECTOR_STATES_DIR, f"{name}.json"))

    @staticmethod
    def save_session(session: ActiveLearningSession) -> None:
        _check_name_contains_valid_chars(session.name)
        session.experiment_state.save(
            _get_experiment_state_path(session.name, session.iteration)
        )

    @staticmethod
    def load_session(session_name: str) -> ActiveLearningSession:
        iteration = FileUtils.find_last_saved_iteration(session_name)
        if iteration is None:
            raise SessionException(
                f"Tried to load session {session_name} but the session does not exist "
                "or the session directory is corrupted.\n"
                f"(No file like experiment_state_*.json found in {_get_session_dir(session_name)})"
            )
        experiment_state = ExperimentState.load(
            _get_experiment_state_path(session_name, iteration)
        )
        session = ActiveLearningSession(
            session_name, experiment_state, iteration=iteration
        )
        return session

    @staticmethod
    def delete_session(session_name: str) -> None:
        try:
            rmtree(_get_session_dir(session_name))
            print(f"Deleted session: {session_name}")
        except:
            pass

    @staticmethod
    def new_session(
        session_name: str, experiment_state: ExperimentState
    ) -> ActiveLearningSession:

        if not _dir_is_empty(_get_session_dir(session_name)):
            raise SessionException(
                f"Cannot create new session with name {session_name} as there already exists "
                "a session with that name. Use FileUtils.delete_session() if you want to remove "
                "the existing session."
            )

        session = ActiveLearningSession(session_name, experiment_state, iteration=1)
        FileUtils.save_session(session)
        print(f"Created session: {session.name}")
        return session

    @staticmethod
    def save_session_model(session: ActiveLearningSession) -> None:
        try:
            param_file = _get_saved_model_params_path(session.name, session.iteration)
            session.model.save(param_file)
            session.experiment_state.model.parameter_file = param_file
            session.experiment_state.model.save(
                _get_saved_model_state_path(session.name, session.iteration)
            )
        except RuntimeWarning as e:
            print(f"Session.save_model: {e}")

    @staticmethod
    def find_last_saved_iteration(session_name: str) -> int:
        experiment_paths = glob.glob(_get_experiment_state_path(session_name, "*"))
        if len(experiment_paths) == 0:
            return None

        def _get_iteration_from_path(name: str):
            no_ext = name.split(".")[0]
            return int(no_ext.split("_")[-1])

        return max([_get_iteration_from_path(path) for path in experiment_paths])

    @staticmethod
    def save_iteration_report(session_name: str, iteration_report: IterationReport):
        iteration_report.save(
            _get_iteration_report_path(
                session_name, iteration=iteration_report.iteration
            )
        )

    @staticmethod
    def load_all_iteration_reports(session_name: str) -> list[IterationReport]:
        report_files = glob.glob(
            join(_get_session_dir(session_name), "iteration_report_*.json")
        )
        if len(report_files) == 0:
            raise SessionException(f"No reports for session {session_name} found!")
        reports = [None] * len(
            report_files
        )  # Necessary because files might be in wrong order
        for fname in report_files:
            it = int(basename(fname).split(".")[0].split("_")[-1])
            reports[it - 1] = IterationReport.load(fname)
        return reports

    @staticmethod
    def load_all_experiment_states(session_name: str) -> list[ExperimentState]:
        state_files = glob.glob(
            join(_get_session_dir(session_name), "experiment_state_*.json")
        )
        if len(state_files) == 0:
            raise SessionException(f"No experiment states for session {session_name} found!")
        states = [None] * len(
            state_files
        )  # Necessary because files might be in wrong order
        for fname in state_files:
            it = int(basename(fname).split(".")[0].split("_")[-1])
            states[it - 1] = ExperimentState.load(fname)
        return states

    @staticmethod
    def get_saved_data_states() -> list[str]:
        """
        Get all exported data states.
        """
        return [
            splitext(f)[0]
            for f in os.listdir(DATA_STATES_DIR)
            if isfile(join(DATA_STATES_DIR, f))
        ]

    @staticmethod
    def get_all_session_names(session_pattern: str) -> Generator[str, None, None]:
        """
        Session names are subdivided by the "." character.
        Every single part of the session name can consist of alphanumeric characters and underscores.
        The * pattern matches any number of any characters within the same subname
        (i.e. the * stops at the next ".")
        """
        def _match_session_pattern(pattern: str, name: str) -> bool:
            for subpat, subname in zip(pattern.split("."), name.split(".")):
                subregex = re.compile(subpat.replace("*", ".*"))
                if not subregex.match(subname):
                    return False
            return True

        # Check the pattern contains only characters that are allowed in a session name or "*"
        _check_name_contains_valid_chars(session_pattern.replace("a", "_"))
        for session_dir, dirnames, filenames in os.walk(SESSIONS_DIR):
            dir_path = relpath(session_dir, start=SESSIONS_DIR)
            if dir_path[0] == ".":
                continue
            if len(filenames) == 0:
                continue
            session_name = dir_path.replace(os.sep, ".")
            if _match_session_pattern(pattern=session_pattern, name=session_name):
                yield session_name
