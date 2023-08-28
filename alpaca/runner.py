"""
CLI for running active learning experiments implemented with ALPACA

(which stands for "Active Learning Pipeline ACronym ALPACA")
"""

"""
                                                           %(/      &(/
                                                           (.#%%%%%%&%&
                                                            &#/####(@#%
                                                           .%####%%@%##
                                                           &##((###%(,/
                                                           #((*/*,(/**.
                                                          .#(#/(/***/%
                                                          %%%#((((((*#
                                                          %###(/(((##,
                   *&&&&@&&&&&&%&/                       ##((/((((#%%
              *@&&%%%%#%##%##%%%%%%%#%####%%&&&&&&@%,   &#((/#(#((#%&
           #&%###%%%#####((%#%((#(##%##############(###(((((/((((##%.
         .&%#/&##(((((((((/((/(((/(((#((((((/(((/##((#(/#////((#####
        .&%((##((##((//*(///(///(#(((((//##**((/((((//#(/*//(/(/(##
        /#(/#%/(/**/*////#/**/.(/((/(////*/**////(//////(///*#/(#
         /#/(#((**((**/*/**(.//**//,***///,**///////(/((/(/##((/
          .(*##(/*//*////**/ ,*//**,*,***/*//*/*///*((((/##/#*
            %%((/*,*,,.*****, .,,*,,,,/**///*,**/*,*////((##
            ,(#/*(*/*,..,,.    *,,,,,,,**,,* ,*,,**/(**/(%/
             %#(**,.                   ..,.  ....,*/***/*
             #(/,*,..,**                 #.. .,(/((*/(#
             %(**(**,/(                  (*,,*/#(/(//##
           *#***/.*//,                   #/,.(*//****/
           #/*,.**(.                     (*., #(,**//,
           /,,../(%                      /#(* ////**%
           .,,.,((#                      ,(/,.,(,*./,
            ,/*//((                      ,(*..  /*,(.
             /*/##/%                     %//,.  #*/*,
              *,*//(%                    %//(,  (*/#(
                 ,*//*.                  ,(\(    .\((\

"""

from typing import Callable, Generator
import argparse
from dataclasses import dataclass
import re

from os.path import dirname, abspath, join

from alpaca.visualization.plot_rmse_history import (
    plot_RMSE_over_iterations,
    PerformanceHistory,
)
from alpaca.system.factory import (
    FactoryUtils,
    DataState,
    ModelState,
    SelectorState,
    ExperimentState,
)
from alpaca.system.files import FileUtils
from alpaca.system.run_experiment import run_experiment
import alpaca.experiments.config as config
from alpaca.experiments._numpy_datasets import (
    NUMPY_DATASETS,
)
from alpaca.experiments.random_selector import get_RandomSelector
from alpaca.experiments.linear_regression import (
    get_LinearRegression,
    get_LassoRegression,
    get_RidgeRegression,
)
from alpaca.experiments.bmdal import (
    get_bmdal_bait_f_p_grad_rp_512_train,
    get_bmdal_fw_p_ll_acs_rf_hyper_512,
    get_bmdal_kmeanspp_p_grad_rp_512_acs_rf_512,
    get_bmdal_lcmd_tp_grad_rp_512,
    get_bmdal_maxdet_p_grad_rp_512_train,
    get_bmdal_maxdiag_grad_rp_512_acs_rf_512,
    get_bmdal_maxdist_p_grad_rp_512_train,
    get_bmdal_random,
)
from alpaca.experiments.dropout import (
    get_DropoutModelSmall,
    get_DropoutModelMedium,
    get_DropoutModelWide,
    get_DropoutModelDeep,
    get_ConcreteDropoutModelSmall,
    get_ConcreteDropoutMedium,
    get_ConcreteDropoutModelWide,
    get_ConcreteDropoutModelDeep,
    get_DropoutSelector,
)
from alpaca.experiments.uncertainty import (
    get_UncertaintyModelSmall,
    get_UncertaintyModelMedium,
    get_UncertaintyModelWide,
    get_UncertaintyModelDeep,
    get_UncertaintySelectorF1,
    get_UncertaintySelectorQuantile,
)
from alpaca.experiments._torch_models import (
    get_TorchModelSmall,
    get_TorchModelMedium,
    get_TorchModelWide,
    get_TorchModelDeep,
)

PLOTS_DIR: str = join(dirname(dirname(abspath(__file__))), "reports", "figures")

DATASETS: dict[str, Callable[[], DataState]] = NUMPY_DATASETS


@dataclass
class ModelTraits:
    # Traits to determine compatibility with selectors
    type: str = None
    # Model architecture
    arch: str = None

    def match_required(self, required: "ModelTraits") -> bool:
        if type(required) != ModelTraits:
            raise RuntimeError()
        for trait, val in required.__dict__.items():
            if val is None:
                continue
            if val != self.__getattribute__(trait):
                return False
        return True


MODELS: dict[str, Callable[[], tuple[Callable[[], ModelState], ModelTraits]]] = {
    "linear_regression": (
        get_LinearRegression,
        ModelTraits(type="sklearn"),
    ),
    "lasso_regression": (
        get_LassoRegression,
        ModelTraits(type="sklearn"),
    ),
    "ridge_regression": (
        get_RidgeRegression,
        ModelTraits(type="sklearn"),
    ),
    "torch_small": (
        get_TorchModelSmall,
        ModelTraits(type="torch", arch="small"),
    ),
    "torch_medium": (get_TorchModelMedium, ModelTraits(type="torch", arch="medium")),
    "torch_wide": (get_TorchModelWide, ModelTraits(type="torch", arch="wide")),
    "torch_deep": (get_TorchModelDeep, ModelTraits(type="torch", arch="deep")),
    "dropout_small": (
        get_DropoutModelSmall,
        ModelTraits(type="dropout", arch="small"),
    ),
    "dropout_medium": (
        get_DropoutModelMedium,
        ModelTraits(type="dropout", arch="medium"),
    ),
    "dropout_wide": (get_DropoutModelWide, ModelTraits(type="dropout", arch="wide")),
    "dropout_deep": (get_DropoutModelDeep, ModelTraits(type="dropout", arch="deep")),
    "concrete_dropout_small": (
        get_ConcreteDropoutModelSmall,
        ModelTraits(type="dropout", arch="small"),
    ),
    "concrete_dropout_medium": (
        get_ConcreteDropoutMedium,
        ModelTraits(type="dropout", arch="medium"),
    ),
    "concrete_dropout_wide": (
        get_ConcreteDropoutModelWide,
        ModelTraits(type="dropout", arch="wide"),
    ),
    "concrete_dropout_deep": (
        get_ConcreteDropoutModelDeep,
        ModelTraits(type="dropout", arch="deep"),
    ),
    "uncertainty_small": (
        get_UncertaintyModelSmall,
        ModelTraits(type="uncertainty", arch="small"),
    ),
    "uncertainty_medium": (
        get_UncertaintyModelMedium,
        ModelTraits(type="uncertainty", arch="medium"),
    ),
    "uncertainty_wide": (
        get_UncertaintyModelWide,
        ModelTraits(type="uncertainty", arch="wide"),
    ),
    "uncertainty_deep": (
        get_UncertaintyModelDeep,
        ModelTraits(type="uncertainty", arch="deep"),
    ),
}
MODEL_CONSTRUCTORS: dict[str, Callable[[], ModelState]] = {
    key: tpl[0] for key, tpl in MODELS.items()
}
MODEL_TRAITS: dict[str, ModelTraits] = {key: tpl[1] for key, tpl in MODELS.items()}

SELECTORS: dict[str, Callable[[], tuple[Callable[[], SelectorState], ModelTraits]]] = {
    "random": (get_RandomSelector, ModelTraits()),
    "dropout": (get_DropoutSelector, ModelTraits(type="dropout")),
    "uncertainty_f1": (get_UncertaintySelectorF1, ModelTraits(type="uncertainty")),
    "uncertainty_quantile": (
        get_UncertaintySelectorQuantile,
        ModelTraits(type="uncertainty"),
    ),
    "bmdal_random": (get_bmdal_random, ModelTraits(type="torch")),
    "bmdal_bald": (get_bmdal_maxdiag_grad_rp_512_acs_rf_512, ModelTraits(type="torch")),
    "bmdal_batch_bald": (
        get_bmdal_maxdet_p_grad_rp_512_train,
        ModelTraits(type="torch"),
    ),
    "bmdal_bait": (get_bmdal_bait_f_p_grad_rp_512_train, ModelTraits(type="torch")),
    "bmdal_acs_fw": (get_bmdal_fw_p_ll_acs_rf_hyper_512, ModelTraits(type="torch")),
    "bmdal_coreset": (get_bmdal_maxdist_p_grad_rp_512_train, ModelTraits(type="torch")),
    "bmdal_badge": (
        get_bmdal_kmeanspp_p_grad_rp_512_acs_rf_512,
        ModelTraits(type="torch"),
    ),
    "bmdal_lcmd": (get_bmdal_lcmd_tp_grad_rp_512, ModelTraits(type="torch")),
}
SELECTOR_CONSTRUCTORS: dict[str, Callable[[], SelectorState]] = {
    key: tpl[0] for key, tpl in SELECTORS.items()
}
SELECTOR_TRAITS: dict[str, ModelTraits] = {
    key: tpl[1] for key, tpl in SELECTORS.items()
}


def _star_pattern_filter(strs: list[str], pattern: str) -> list[str]:
    """
    Putting a start in a split, model, selector or dataset key matches any number of characters
    """
    regex = re.compile(pattern.replace("*", ".*"))
    return [s for s in strs if regex.match(s)]


def _datasets_info() -> None:
    print(f"\nAvailable datasets ({len(DATASETS)}): ")
    for k in DATASETS.keys():
        print(f"    {k}")


def _splits_info() -> None:
    splits = FileUtils.get_saved_data_states()
    print(f"\nAvailable train/test/pool splits of datasets ({len(splits)}): ")
    for s in splits:
        print(f"    {s}")


def _experiments_info() -> None:
    print(f"\nAvailable experiments ({len(SELECTORS)}): ")

    for selector, traits in SELECTOR_TRAITS.items():
        models = [
            m
            for m, model_traits in MODEL_TRAITS.items()
            if model_traits.match_required(traits)
        ]
        print(f"    {selector} supporting {len(models)} model variant(s)")
        for m in models:
            print(f"        - {m}")


def _sessions_info() -> None:
    sessions = list(FileUtils.get_all_session_names(".*"))
    print(f"\nExisting sessions ({len(sessions)}): ")
    for s in sessions:
        print(f"    {s}")


def info(datasets_flag: bool, splits_flag: bool, experiments_flag: bool, sessions_flag: bool) -> None:
    if not (datasets_flag or splits_flag or experiments_flag or sessions_flag):
        _datasets_info()
        _splits_info()
        _experiments_info()
        _sessions_info()
        return
    if datasets_flag:
        _datasets_info()
    if splits_flag:
        _splits_info()
    if experiments_flag:
        _experiments_info()
    if sessions_flag:
        _sessions_info()


def split(
    name: str,
    dataset: str,
    train: float,
    pool: float,
    test: float,
    seed: int,
) -> None:
    data_state: DataState = DATASETS[dataset]()
    dataset_split = FactoryUtils.random_split_dataset(
        data_state.dataset_key,
        dataset_args=data_state.dataset_args,
        train_pool_test_split=(train, pool, test),
        seed=seed,
    )
    FileUtils.save_data_state(dataset_split, name)


def _run_with_split_and_selector(
    split_key: str, selector_key: str, model: str, new_flag: bool
) -> None:
    data_state = FileUtils.load_data_state(split_key)
    selector_state = SELECTOR_CONSTRUCTORS[selector_key]()

    model_keys = _star_pattern_filter(MODELS.keys(), model)
    if len(model_keys) == 0:
        print(f"No model keys match argument {model}")

    model_keys = [
        k
        for k in model_keys
        if MODEL_TRAITS[k].match_required(SELECTOR_TRAITS[selector_key])
    ]
    if len(model_keys) == 0:
        print(
            f"No models which match argument {model} are compatible with selector {selector_key}"
        )

    for model_key in model_keys:
        name = f"{split_key}.{selector_key}.{model_key}"
        model_state = MODEL_CONSTRUCTORS[model_key]()
        experiment_state = ExperimentState(
            data=data_state, model=model_state, selector=selector_state, seed=config.SEED
        )
        run_experiment(
            name=name,
            experiment_state=experiment_state,
            batch_size=config.QUERY_BATCH_SIZE,
            max_iteration=config.MAX_NUM_QUERIES,
            new_experiment=new_flag,
            save_model = config.SAVE_MODELS
        )
        yield name


def _run_with_split(split_key: str, selector: str, model: str, new_flag: bool):
    selector_keys = _star_pattern_filter(SELECTORS.keys(), selector)
    if len(selector_keys) == 0:
        print(f"No selectors match argument {selector}!")
    for selector_key in selector_keys:
        for experiment_name in _run_with_split_and_selector(split_key, selector_key, model, new_flag):
            yield experiment_name


def run(
    dataset_split: str,
    selector: str,
    model: str,
    new_flag: bool) -> Generator[list[str], None, None]:
    """
    Runs all possible combinations of split, selector and model that match the provided patterns.
    If the new_flag is set, any existing experiment states for the given experiments are deleted and
    experiments are started at iteration 1.

    yields the names of all experiments that were run.
    """
    split_keys = FileUtils.get_saved_data_states()
    split_keys = _star_pattern_filter(split_keys, dataset_split)
    if len(split_keys) == 0:
        print(f"No dataset splits match argument {dataset_split}!")
    for split_key in split_keys:
        for experiment_name in _run_with_split(split_key, selector, model, new_flag):
            yield experiment_name


def plot(
    name: str,
    sessions: list[str],
    train_flag: bool,
    pool_flag: bool,
    test_flag: bool,
) -> None:
    hists: list[PerformanceHistory] = []
    session_names = [
        session
        for pattern in sessions
        for session in FileUtils.get_all_session_names(pattern)
    ]
    if len(session_names) < 1:
        print(f"No session found that match {sessions}")
        exit(1)
    plotall = not (train_flag or pool_flag or test_flag)
    for session in session_names:
        iteration_reports = FileUtils.load_all_iteration_reports(session)
        if plotall or train_flag:
            hists.append(
                PerformanceHistory(
                    [r.performance.train_rmse for r in iteration_reports],
                    f"{session} : train_rmse",
                )
            )
        if plotall or pool_flag:
            hists.append(
                PerformanceHistory(
                    [r.performance.pool_rmse for r in iteration_reports],
                    f"{session} : pool_rmse",
                )
            )
        if plotall or test_flag:
            hists.append(
                PerformanceHistory(
                    [r.performance.test_rmse for r in iteration_reports],
                    f"{session} : test_rmse",
                )
            )
    plot_RMSE_over_iterations(hists, join(PLOTS_DIR, name))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="CLI for running experiments using ALPACA",
    )

    # INFO
    parser_info = subparsers.add_parser(
        "info", help="Print info on available datasets, selectors and sessions"
    )
    parser_info.add_argument(
        "--datasets", action="store_true", help="Display available datasets"
    )
    parser_info.add_argument(
        "--splits", action="store_true", help="Display created train, test, pool splits"
    )
    parser_info.add_argument(
        "--experiments", action="store_true", help="Display experiments that can be run"
    )
    parser_info.add_argument(
        "--sessions", action="store_true", help="Display existing sessions"
    )

    # RUN
    parser_run = subparsers.add_parser(
        "run",
        help="Start or continue an active learning experiment with the given data state",
    )
    parser_run.add_argument(
        "dataset_split",
        type=str,
        help="The name of the dataset split to use. '*' to run for all splits",
    )
    parser_run.add_argument(
        "selector",
        type=str,
        default="*",
        nargs="?",
        help="The name of the selector to use. '*' to run for all selectors. Default: '*'",
    )
    parser_run.add_argument(
        "model",
        type=str,
        nargs="?",
        default="*",
        help="Model type to use with the selector. '*' to use all compatible models. Default: '*'",
    )
    parser_run.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Start a new experiment, any existing savefiles will be deleted",
    )

    # SPLIT
    parser_testgen = subparsers.add_parser(
        "split", help="Creates a new train/test/pool split of the specified dataset"
    )
    parser_testgen.add_argument(
        "name",
        type=str,
        help="The name of the new dataset split",
    )
    parser_testgen.add_argument(
        "dataset",
        type=str,
        help="The name of the dataset for which to create the split",
    )
    parser_testgen.add_argument(
        "train",
        type=float,
        help="Number or proportion of samples in train set",
    )
    parser_testgen.add_argument(
        "pool",
        type=float,
        help="Numper or proportion of samples in pool set",
    )
    parser_testgen.add_argument(
        "test",
        type=float,
        help="Number or proportion of samples in test set",
    )
    parser_testgen.add_argument(
        "seed",
        type=int,
        nargs="?",
        default=1337,
        help="The seed for python's random",
    )

    # PLOT
    parser_plot = subparsers.add_parser("plot", help="Plots results of an experiment")
    parser_plot.add_argument(
        "name", type=str, help="Name of the plot without file extension"
    )
    parser_plot.add_argument(
        "sessions",
        type=str,
        nargs="+",
        help="The names of the sessions to plot. You can use the * character",
    )
    parser_plot.add_argument(
        "--train",
        action="store_true",
        help="Plot train rmse",
    )
    parser_plot.add_argument(
        "--pool",
        action="store_true",
        help="Plot pool rmse",
    )
    parser_plot.add_argument(
        "--test",
        action="store_true",
        help="Plot test rmse",
    )

    args = parser.parse_args()
    if args.command == "info":
        info(
            args.datasets,
            args.splits,
            args.sessions,
            args.experiments)
        exit(0)
    elif args.command == "run":
        run(
            args.dataset_split,
            args.selector,
            args.model,
            args.new
        )
        exit(0)
    elif args.command == "split":
        split(
            args.name,
            args.dataset,
            args.train,
            args.pool,
            args.test,
            args.seed,
        )
        exit(0)
    elif args.command == "plot":
        plot(
            args.name,
            args.sessions,
            args.train,
            args.pool,
            args.test)
        exit(0)


if __name__ == "__main__":
    main()
