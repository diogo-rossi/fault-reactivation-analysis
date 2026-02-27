import types
from typing import Any, Literal, cast

from numpy import float64
from numpy.typing import NDArray
from plotly.graph_objects import Figure
from streamlit import session_state as ss
from streamlit.runtime.state import SessionStateProxy

from figure_deterministic import DeterministicAnalisisFigure
from figure_probabilistic import ProbabilisticAnalisisFigure
from tables_types import LayerTable

Keys = Literal[
    "layer_names",
    "final_layers_df",
    "inj_top",
    "inj_bas",
    "run_calcs",
    "layer_slider_value",
    "dP_slider_value",
    "fig2",
    "fig3",
    "SFs",
    "fp",
    "bins",
]


class SessionState(SessionStateProxy):
    layer_names: list[str]
    final_layers_df: LayerTable
    inj_top: float
    inj_bas: float
    run_calcs: bool
    layer_slider_value: int
    dP_slider_value: float
    fig2: DeterministicAnalisisFigure | None
    fig3: ProbabilisticAnalisisFigure | None
    SFs: NDArray[float64] | None
    fp: NDArray[float64] | None
    bins: int

    def __getitem__(self, n) -> Any: ...

    def get(self, n: Keys, default: Any | None = None):
        return super().get(n, default)


def init_key(key: Keys, value: Any):
    if key not in ss:
        ss[key] = value


ss = cast(SessionState, ss)
