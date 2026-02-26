import streamlit as st

col_conf_injection_layer = {
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

col_conf_layers = {
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
