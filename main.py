from __future__ import annotations
import os
import streamlit as st
import streamlitrunner as sr
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from typing import overload, Literal, Any, cast
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter, Contour, Histogram, Figure
from plotly.graph_objs.scatter import Line
from scipy.stats import truncnorm, percentileofscore
from streamlit import session_state as ss

# os.system("cls")

VERTICAL_DIVISIONS: int = 1000

gamma: NDArray[float64] = np.zeros((VERTICAL_DIVISIONS, 1))


# fmt: off
fig = make_subplots(
    rows=2,
    cols=4,
    specs=[
        [{"rowspan": 2}, {"rowspan": 2}, {"rowspan": 2}, {"colspan": 1}],
        [          None,           None,           None, {"colspan": 1}],
    ],
    horizontal_spacing=0.01,
    vertical_spacing=0.07,
)
# fmt: on

axeskwargs = {
    "mirror": "allticks",
    "ticks": "inside",
    "showgrid": True,
    "showline": True,
    "showticklabels": True,
    "title_standoff": 2,
}

# %%          FUNCTIONS
############# FUNCTIONS ################################################################


def FSfun(gammas, thicknesses, inj_id):
    """TODO"""
    gammas[:inj_id] * thicknesses[:inj_id]


def dist(
    number_of_values: int,
    distribution_choice: Literal["Uniform", "Triangular", "Normal"],
    mean: float,
    std_dev: float,
    lower_value: float,
    upper_value: float,
):
    """Return a distribution based on statistics"""
    a = (lower_value - mean) / std_dev
    b = (upper_value - mean) / std_dev
    if distribution_choice == "Uniform":
        return np.random.uniform(lower_value, upper_value, number_of_values)
    elif distribution_choice == "Triangular":
        return np.random.triangular(lower_value, mean, upper_value, number_of_values)
    elif distribution_choice == "Normal":
        return truncnorm.rvs(a, b, mean, std_dev, number_of_values)


def line(
    row: int, col: int, value: float, figure: Figure, horizontal: bool = False, **kwargs
):
    """Adds a line to `fig`, horizontal or vertical"""
    if horizontal:
        figure.add_hline(value, row, col, line_color="black", line_width=2, **kwargs)  # type: ignore
    else:
        figure.add_vline(value, row, col, line_color="black", line_width=2, **kwargs)  # type: ignore


def plot(
    row: int,
    col: int,
    x: NDArray[float64] | list[float],
    y: NDArray[float64] | list[float],
    name: str,
    figure: Figure,
    color: str = "black",
    width: int = 1,
    showlegend: bool = True,
    **kwargs,
):
    """Adds a Scatter plot to `fig`"""
    figure.add_trace(
        row=row,
        col=col,
        trace=Scatter(
            x=x,
            y=y,
            name=name,
            line=Line(color=color, width=width),
            showlegend=showlegend,
            **kwargs,
        ),
    )


def plot_mohr_circle(
    fig: Figure,
    sigma1: float,
    sigma3: float,
    angle: float,
    color: str = "black",
    label: str = "Mohr Circle",
):
    """Plot the upper half of Mohr's circle in `fig[1,4]` with shear plane point"""
    center = (sigma1 + sigma3) / 2
    radius = (sigma1 - sigma3) / 2

    theta = np.linspace(0, np.pi, 100)
    x = center + radius * np.cos(theta)
    y = radius * np.sin(theta)

    plot(1, 4, x, y, name=label, color=color, showlegend=False, figure=fig)

    theta_rad = np.radians(2 * angle)
    sigma_n = center - radius * np.cos(theta_rad)
    tau = radius * np.sin(theta_rad)
    plot(
        1,
        4,
        np.array([center, sigma_n]),
        np.array([0, tau]),
        name="Stress on fault",
        showlegend=True,
        line_dash="dash",
        figure=fig,
    )


# %%          CLASSES
############# CLASSES ##################################################################


class InjectionLayerTable(DataFrame):
    parameter: Series[str]
    distr: Series[str]
    average: Series[float]
    stddevi: Series[float]
    minimum: Series[float]
    maximum: Series[float]

    @overload
    def __getitem__(self, n: Literal["parameter"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["distr"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["average"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["stddevi"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["minimum"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["maximum"]) -> Series[float]: ...

    def __getitem__(self, n) -> Any:
        return super().__getitem__(n)

    def get_value_from_df(
        self,
        line_name: Literal[
            "Ko", "Ka", "biot", "cohesion", "friction", "inclination", "overpressure"
        ],
        column_name: Literal["average", "stddevi", "minimum", "maximum"],
    ):
        return self[column_name][
            [line_name.lower() in line.lower() for line in self.iloc[:, 0]]
        ].to_list()[0]

    def get_mean_value_from_df(
        self,
        line_name: Literal[
            "Ko", "Ka", "biot", "cohesion", "friction", "inclination", "overpressure"
        ],
    ):
        return self.get_value_from_df(line_name, "average")


class LayerTable(DataFrame):
    layer: Series[str]
    depth: Series[float]
    distr: Series[str]
    average: Series[float]
    stddevi: Series[float]
    minimum: Series[float]
    maximum: Series[float]

    @overload
    def __getitem__(self, n: Literal["layer"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["depth"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["distr"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["average"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["stddevi"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["minimum"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["maximum"]) -> Series[float]: ...

    def __getitem__(self, n) -> Any:
        return super().__getitem__(n)


from streamlit.runtime.state import SessionStateProxy


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

# %%          FUNCTIONS CALLBACK
############# FUNCTIONS CALLBACK #######################################################


# %%          INITIAL DATA
############# INITIAL DATA #############################################################

injection_layer_df = DataFrame(
    {
        "parameter": [
            "Initial lateral stress multiplier - Ko [-]",
            "Active lateral stress multiplier - Ka [-]",
            "Biot coefficient - α [-]",
            "Fault cohesion - c [MPa]",
            "Fault friction angle - ϕ [°]",
            "Fault inclination angle - θ [°]",
        ],
        "distr": ["Normal", "Normal", "Normal", "Normal", "Normal", "Normal"],
        "average": [0.25, 1.00, 0.50, 2, 30.0, 25.0],
        "stddevi": [0.05, 0.10, 0.10, 0.1, 5, 5],
        "minimum": [0.05, 0.05, 0.10, 1, 5, 10],
        "maximum": [0.70, 1.70, 1.00, 5, 50, 45],
    },
)

layers_df = DataFrame(
    {
        "layer": [
            "First layer name",
            "Second layer name",
            "Third layer name",
            "Fourth layer name",
            "Fifth layer name",
        ],
        "depth": [100, 200, 800, 1800, 2000],
        "distr": ["Normal", "Normal", "Normal", "Normal", "Normal"],
        "average": [24, 25, 23, 25, 25],
        "stddevi": [2, 2, 2, 2, 2],
        "minimum": [20, 20, 20, 20, 20],
        "maximum": [27, 27, 27, 27, 27],
    },
)

column_config_injection_layer_df = {
    "layer": st.column_config.Column(label="Parameter"),
    "distr": st.column_config.SelectboxColumn(
        label="Distribution", options=["Normal", "Triangular", "Uniform"]
    ),
    "average": st.column_config.NumberColumn(
        label="Mean value",
        format="%.2f",
        min_value=0,
        step=0.01,
    ),
    "stddevi": st.column_config.NumberColumn(
        label="Standard deviation", min_value=0.00, format="%.2f"
    ),
    "minimum": st.column_config.NumberColumn(
        label="Minimum value", min_value=0.00, format="%.2f"
    ),
    "maximum": st.column_config.NumberColumn(
        label="Maximum value", min_value=0.00, format="%.2f"
    ),
}

column_config_layers_df = {
    "layer": st.column_config.Column(label="Layer name"),
    "depth": st.column_config.NumberColumn(
        label="Layer bottom depth [m]", min_value=0.00, format="%.2f"
    ),
    "distr": st.column_config.SelectboxColumn(
        label="Distribution", options=["Normal", "Triangular", "Uniform"]
    ),
    "average": st.column_config.NumberColumn(
        label="Mean value",
        format="%.2f",
        min_value=0,
        step=0.01,
    ),
    "stddevi": st.column_config.NumberColumn(
        label="Standard deviation", min_value=0.00, format="%.2f"
    ),
    "minimum": st.column_config.NumberColumn(
        label="Minimum value", min_value=0.00, format="%.2f"
    ),
    "maximum": st.column_config.NumberColumn(
        label="Maximum value", min_value=0.00, format="%.2f"
    ),
}

# %%          MAIN FUNCTION
############# MAIN FUNCTION ############################################################


def main():
    st.set_page_config(layout="wide")
    st.markdown(
        """
    <style>
        footer {visibility: hidden;}

        /* Remove top padding from main container */
        .block-container {
            padding-bottom: 0rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    global fig

    if ss.get("run_calcs") is None:
        ss.run_calcs = True

    if ss.get("layer_names") is None:
        ss.layer_names = layers_df["layer"].to_list()

    if ss.get("layer_slider_value") is None:
        ss.layer_slider_value = 460

    if ss.get("dP_slider_value") is None:
        ss.dP_slider_value = 1

    if ss.get("figure_tab2") is None:
        ss.figure_tab2 = None

    if ss.get("bins") is None:
        ss.bins = 50

    tab1, tab2, tab3 = st.tabs(
        [
            "Parameters",
            "Results: deterministic analisis",
            "Results: probabilistic analisis",
        ]
    )

    ####################################################################################
    # %           FIRST TAB
    ####################################################################################

    def set_run_calcs():
        ss.run_calcs = True

    cols = tab1.columns([13, 5, 4, 4, 4, 5, 4, 3])
    cols[0].header("Parameters of the injection layer")
    dPres = cols[1].number_input(value=0.0, label="Initial overpressure [MPa]")
    dPmax = cols[2].number_input(value=5.0, label="Maximum ΔP [MPa]")
    numDP = cols[3].number_input(value=50, label="Number of steps")
    gamaW = cols[4].number_input(value=10.0, label="Water gradient [kN/m³]")
    listcontainer = cols[5].container()

    Nrel = cols[6].number_input("Number of realizations", value=100000)
    cols[7].button("Run", on_click=set_run_calcs)

    ## DATAFRAMES

    final_injection_layer_df = InjectionLayerTable(
        tab1.data_editor(
            injection_layer_df,
            column_config=column_config_injection_layer_df,
            hide_index=True,
        )
    )

    tab1.header("Layer depths and Specific Weight - γ [kN/m³]")
    final_layers_df = LayerTable(
        tab1.data_editor(
            layers_df,
            column_config=column_config_layers_df,
            hide_index=True,
            num_rows="dynamic",
        )
    )

    if None in final_layers_df.iloc[-1, :].to_list():
        tab1.text("Finish input data")
        return

    ss.layer_names = final_layers_df["layer"].to_list()

    with listcontainer:
        inj_lay_name = st.selectbox("Injection layer", ss.layer_names, index=3)

    if "inj_top" not in ss:
        ss.inj_top = final_layers_df.depth[3]
    if "inj_bas" not in ss:
        ss.inj_bas = final_layers_df.depth[4]

    if None not in final_layers_df.depth:
        inj_layer_pos = (
            (final_layers_df.iloc[:, 0] == inj_lay_name).to_list().index(True)
        )
        ss.inj_top = (
            0.0 if inj_layer_pos == 0 else final_layers_df.depth[inj_layer_pos - 1]
        )
        ss.inj_bas = final_layers_df.depth[inj_layer_pos]

    inj_top = ss.inj_top
    inj_bas = ss.inj_bas

    fig = ss.figure_tab2 or fig

    ####################################################################################
    # %           GETTING PARAMETERS
    ####################################################################################

    depths = final_layers_df.depth.to_numpy()
    depths = np.insert(depths, 0, 0)
    max_depth = depths.max()
    z = np.linspace(0, max_depth, VERTICAL_DIVISIONS)
    inj_pos = (inj_top <= z) & (z <= inj_bas)
    dz = np.insert(np.diff(z), 0, 0)[:, None]

    for i in final_layers_df.index:
        gamma[(depths[i] <= z) & (z <= depths[i + 1])] = final_layers_df.average[i]

    Ko = final_injection_layer_df.get_mean_value_from_df("Ko")
    Ka = final_injection_layer_df.get_mean_value_from_df("Ka")
    biot = final_injection_layer_df.get_mean_value_from_df("biot")
    coh = final_injection_layer_df.get_mean_value_from_df("cohesion") * 1000
    phi = np.radians(final_injection_layer_df.get_mean_value_from_df("friction"))
    theta = np.radians(final_injection_layer_df.get_mean_value_from_df("inclination"))

    # MPa to kPa
    dPmax *= 1000
    dPres *= 1000
    numDP += 1

    dP = np.linspace(0, dPmax, numDP)

    ####################################################################################
    # %           CALCULATION
    ####################################################################################

    Pp = gamaW * z[:, None] * np.ones(numDP)
    FSres = np.zeros(VERTICAL_DIVISIONS)[:, None] * np.ones(numDP)

    Pp[inj_pos, :] = Pp[inj_pos, :] + dPres
    for i in range(numDP):
        Pp[inj_pos, i] = Pp[inj_pos, i] + dP[i]

    SvTotal = np.cumsum(gamma * dz)
    SvEff = SvTotal[:, None] - biot * Pp
    ShEff = Ko * SvEff
    for i in range(1, numDP):
        ShEff[inj_pos, i] = ShEff[inj_pos, 0] - Ka * biot * dP[i]
    ShTotal = ShEff + biot * Pp
    Tn = np.abs(((ShEff - SvEff) / 2) * np.sin(2 * theta))
    Sn = ShEff * (np.cos(theta)) ** 2 + SvEff * (np.sin(theta)) ** 2
    Ts = coh + Sn * np.tan(phi)
    FSres[inj_pos, :] = Ts[inj_pos, :] / Tn[inj_pos, :]
    FSdet = FSres[inj_pos, :]

    SvTotal /= 1000
    SvEff /= 1000
    ShEff /= 1000
    ShTotal /= 1000
    Pp /= 1000
    Sn /= 1000
    Tn /= 1000
    Ts /= 1000
    coh /= 1000
    dPmax /= 1000

    dP /= 1000

    camada = VERTICAL_DIVISIONS - ss.layer_slider_value
    dP_slider = ss.dP_slider_value

    # dP_slider *= 1000
    dPstep: int = np.where(dP >= dP_slider if dPmax > 0 else dP <= dP_slider)[0][0]

    ####################################################################################
    # %           BUILD FIGURE
    ####################################################################################

    for i in range(3):
        fig.update_xaxes(
            row=1,
            col=i + 1,
            matches="x",
            title_text=f"Principal stresses and pore pressure [MPa]",
            **axeskwargs,
        )
        fig.update_yaxes(
            row=1,
            col=i + 1,
            matches="y",
            range=[max_depth, 0],
            **(axeskwargs | {"showticklabels": False}),
        )
    fig.update_yaxes(
        row=1,
        col=1,
        matches="y",
        title_text="Depth [m]",
        showticklabels=True,
        title_standoff=2,
    )

    # ----------------
    row, col = 1, 1

    SvTotal_name = "σ<sub>v</sub> Total"
    SvEff_name = "σ'<sub>v</sub> Effective"
    ShTotal_name = "σ<sub>h</sub> Total"
    ShEff_name = "σ'<sub>h</sub> Effective"
    Pp_name = "ΔP"

    plot(row, col, y=z, x=SvTotal, name=SvTotal_name, color="red", figure=fig)
    plot(
        row,
        col,
        y=z,
        x=SvEff[:, dPstep],
        name=SvEff_name,
        color="brown",
        figure=fig,
    )
    plot(
        row,
        col,
        y=z,
        x=ShTotal[:, dPstep],
        name=ShTotal_name,
        color="green",
        figure=fig,
    )
    plot(
        row,
        col,
        y=z,
        x=ShEff[:, dPstep],
        name=ShEff_name,
        color="cornflowerblue",
        figure=fig,
    )
    plot(row, col, y=z, x=Pp[:, dPstep], name=Pp_name, color="blue", figure=fig)
    for d in depths:
        line(row, col, d, horizontal=True, figure=fig)

    # ----------------
    row, col = 1, 2
    fig.update_xaxes(
        row=row,
        col=col,
        matches="x2",
        title_text="Stresses on fault [MPa]",
        showticklabels=True,
        range=[0, np.max(list(map(np.max, [Tn, Ts, Sn])))],
    )

    Sn_name = "σ<sub>n</sub> Normal"
    Tn_name = "𝜏<sub>n</sub> Shear"
    Ts_name = "𝜏<sub>s</sub> Limit"

    plot(row, col, y=z, x=Sn[:, dPstep], name=Sn_name, color="black", figure=fig)
    plot(row, col, y=z, x=Tn[:, dPstep], name=Tn_name, color="magenta", figure=fig)
    plot(row, col, y=z, x=Ts[:, dPstep], name=Ts_name, color="chocolate", figure=fig)
    for d in depths:
        line(row, col, d, horizontal=True, figure=fig)

    # ----------------
    row, col = 1, 3
    fig.update_xaxes(
        row=row,
        col=col,
        matches="x3",
        title_text="Security Factor (SF)",
        range=[0, np.max(FSres)],
        **axeskwargs,
    )
    plot(
        row,
        col,
        y=z,
        x=FSres[:, dPstep],
        name="SF",
        color="black",
        showlegend=False,
        figure=fig,
    )

    line(row, col, 1.0, horizontal=False, figure=fig)

    # ----------------
    row, col = 1, 4
    Smax = np.max(Sn)
    Tmax = coh + Smax * np.tan(phi)
    fig.update_xaxes(
        row=row,
        col=col,
        range=[0, Smax],
        title_text=f"σ<sub>n</sub> - Normal stress [MPa]",
        **axeskwargs,
    )
    fig.update_yaxes(
        row=row,
        col=col,
        range=[0, Tmax],
        title_text=f"𝜏<sub>n</sub> - Shear stress [MPa]",
        side="right",
        **axeskwargs,
    )
    plot(
        row,
        col,
        [0, Smax],
        [coh, Tmax],
        "Evelope",
        showlegend=False,
        color="red",
        figure=fig,
    )
    plot_mohr_circle(
        fig,
        SvEff[camada, dPstep],
        ShEff[camada, dPstep],
        color="green",
        angle=np.degrees(theta),
    )

    # ----------------
    row, col = 2, 4
    fig.update_xaxes(
        row=row,
        col=col,
        title_text=f"Pressure variation - ΔP [MPa]",
        range=[dP.min(), dP.max()],
        **axeskwargs,
    )
    fig.update_yaxes(
        row=row,
        col=col,
        title_text=f"Injection layer depth [m]",
        side="right",
        range=[z[inj_pos][-1], z[inj_pos][0]],
        **axeskwargs,
    )
    contour = Contour(
        x=dP,
        y=z[inj_pos],
        z=FSdet,
        colorscale="Turbo",
        ncontours=20,
        contours=dict(
            coloring="fill",
            showlines=False,
        ),
        colorbar=dict(
            x=1.05,
            y=0.23,
            len=0.5,
            title=dict(text="Safety Factor (SF)", side="right"),
        ),
        name="SF map",
        showlegend=False,
    )

    contour_line = Contour(
        x=dP,
        y=z[inj_pos],
        z=FSdet,
        contours=dict(
            start=1.0,
            end=1.0,
            size=0.001,
            coloring="none",
            showlabels=True,
            labelformat=".1f",
            labelfont=dict(size=16, color="black"),
        ),
        line=dict(color="black", width=2),
        showscale=False,
        coloraxis=None,
        colorscale=None,
        name="SF map",
        showlegend=False,
    )

    X, Y = np.meshgrid(dP, z[inj_pos][0 : -1 : int(len(z[inj_pos]) / 50)])
    mesh = Scatter(
        x=X.flatten(),
        y=Y.flatten(),
        mode="markers",
        marker=dict(size=5, color="rgba(0,0,0,0)"),
        visible=True,
        showlegend=False,
        name="ΔP vs z",
    )

    fig.add_trace(contour, row=row, col=col)
    fig.add_trace(contour_line, row=row, col=col)
    fig.add_trace(mesh, row=row, col=col)

    fig = fig.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        height=640,
        margin=dict(l=50, r=40, t=30, b=0),
        showlegend=True,
        dragmode="zoom",
    )

    # ss.figure_tab2 = fig

    ####################################################################################
    # %
    ####################################################################################

    figure = fig  # Figure(ss.figure_tab2)

    dP_slider_container = tab2.container()
    cols = tab2.columns([1, 35])
    layer_slider_container = cols[0].container()

    line(1, 1, z[camada], horizontal=True, line_dash="dash", figure=figure)
    line(1, 2, z[camada], horizontal=True, line_dash="dash", figure=figure)
    line(1, 3, z[camada], horizontal=True, line_dash="dash", figure=figure)
    line(2, 4, z[camada], horizontal=True, figure=figure)
    line(2, 4, dP[dPstep], horizontal=False, figure=figure)
    plot(
        2, 4, [dP[dPstep]], [z[camada]], name="Current", showlegend=False, figure=figure
    )

    # ss.figure_tab2

    event = cols[1].plotly_chart(
        figure, theme=None, on_select="rerun", selection_mode="points"
    )
    if event and event["selection"] and event["selection"]["points"]:
        ss.dP_slider_value = event["selection"]["points"][0]["x"]
        ss.layer_slider_value = VERTICAL_DIVISIONS - np.searchsorted(
            z, event["selection"]["points"][0]["y"]
        )
        st.rerun()

    with dP_slider_container:
        st.slider(
            "Pressure step - ΔP [MPa]",
            min_value=0.0 if dPmax > 0 else dPmax,
            max_value=dPmax if dPmax > 0 else 0.0,
            step=np.abs(dPmax / (numDP - 1)),
            # value=1.00,
            key="dP_slider_value",
        )

    with layer_slider_container:
        from streamlit_vertical_slider import vertical_slider

        slider_value = vertical_slider(
            label="Layer",
            min_value=1,
            max_value=VERTICAL_DIVISIONS,
            step=1,
            height=580,
            value_always_visible=False,
            default_value=int(ss.layer_slider_value),
            key=f"layer_slider_value{ss.layer_slider_value}",
        )

        if slider_value != ss.layer_slider_value:
            ss.layer_slider_value = slider_value
            st.rerun()

    ####################################################################################
    # %           GRAPHS ON THIRD TAB
    ####################################################################################

    cols = tab3.columns([6, 1, 4])
    cols[0].header("Histogram distribution of the injection layer parameters")
    nbins = cols[1].number_input("Number of bins", value=ss.bins)
    ss.bins = nbins
    cols[2].header("Analisis results")
    fig_prob = make_subplots(
        rows=2,
        cols=4,
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )
    rows = [1, 1, 1, 2, 2, 2]
    cols = [1, 2, 3, 1, 2, 3]
    for i in final_injection_layer_df.index:

        data = final_injection_layer_df.iloc[i].to_list()
        hist = dist(Nrel, *data[1:])
        fig_prob.update_yaxes(
            row=rows[i],
            col=cols[i],
            title_text=f"Frequency",
            **axeskwargs,
        )

        fig_prob.update_xaxes(
            row=rows[i],
            col=cols[i],
            title_text=data[0],
            **axeskwargs,
        )
        fig_prob.add_trace(
            Histogram(
                x=hist,
                nbinsx=nbins,
                name=data[0],
                marker=dict(line=dict(width=0)),
            ),
            row=rows[i],
            col=cols[i],
        )
        fig_prob.add_annotation(
            text=f"Mean = {data[2]:.2f}<br>"
            f"Std = {data[3]:.2f}<br>"
            f"Min = {data[4]:.2f}<br>"
            f"Max = {data[5]:.2f}<br>"
            f"Samples = {Nrel}",
            xref="x2 domain",
            yref="y2 domain",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
            row=rows[i],
            col=cols[i],
        )

    fig_prob.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        height=650,
        margin=dict(l=50, r=40, t=30, b=0),
        showlegend=True,
    )
    tab3.plotly_chart(fig_prob, theme=None)

    ss.run_calcs = False


if __name__ == "__main__":
    # main()
    sr.run(main, screen=1, open_as_app=True, fill_page_content=False)
