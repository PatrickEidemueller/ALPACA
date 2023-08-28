from alpaca.system.factory import (
    SelectorState,
    SelectorFactory,
)
from alpaca.system.utils.random_selector import RandomSelector

from alpaca.experiments import config

SelectorFactory.register("RandomSelector", RandomSelector)


def get_RandomSelector() -> SelectorState:
    return SelectorState(
        selector_key="RandomSelector", selector_args={"seed": config.SEED}
    )
