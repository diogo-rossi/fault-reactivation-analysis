from typing import Any, Literal, cast

from plotly.graph_objects import Figure
from streamlit import session_state as ss
from streamlit.runtime.state import SessionStateProxy

from tables_types import LayerTable


class SessionState(SessionStateProxy):
    layer_names: list[str]
    final_layers_df: LayerTable
    inj_top: float
    inj_bas: float
    run_calcs: bool
    layer_slider_value: int
    dP_slider_value: float
    figure_tab2: Figure | None
    bins: int

    def __getitem__(self, n) -> Any: ...

    def get(
        self,
        n: Literal[
            "layer_names",
            "final_layers_df",
            "inj_top",
            "inj_bas",
            "run_calcs",
            "layer_slider_value",
            "dP_slider_value",
            "figure_tab2",
            "bins",
        ],
        default: Any | None = None,
    ):
        return super().get(n, default)


ss = cast(SessionState, ss)
