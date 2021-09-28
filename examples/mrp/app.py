import pandas as pd

import streamlit as st

from modules.app_utils import get_data, get_plotting_data, plot_post_strat

PP_COLUMNS = [
    f'support_sample_{sample}' for sample in range(2000)
]

###############################################################################

ps_df, ds_df = get_data(
    ps_path='results\\post_stratified.csv',
    ds_path='results\\agg_polls.csv'
)

##############################################################################
st.sidebar.header('Posterior Samples')
n_samples = st.sidebar.number_input(
    'Posterior Samples Shown',
    min_value=30,
    max_value=2000,
    value=200
)
st.sidebar.header('Demographics Filters')
ethnicity = st.sidebar.multiselect(
    'Etnicity',
     ps_df['black'].unique(),
     ps_df['black'].unique()
)
gender = st.sidebar.multiselect(
    'Gender',
     ps_df['female'].unique(),
     ps_df['female'].unique()
)
edu = st.sidebar.multiselect(
    'Education',
     ps_df['edu'].unique(),
     ps_df['edu'].unique()
)
age = st.sidebar.multiselect(
    'Age',
     ps_df['age'].unique(),
     ps_df['age'].unique()
)

filtered_ps_df = ps_df.copy()
filtered_ds_df = ds_df.copy()
selected_values = {
    'black': ethnicity,
    'female': gender,
    'edu': edu,
    'age': age
}

for column, values in selected_values.items():

    filtered_ds_df = filtered_ds_df[
        filtered_ds_df[column].isin(values)
    ]
    filtered_ps_df = filtered_ps_df[
        filtered_ps_df[column].isin(values)
    ]

###############################################################################

state_ps_df, state_ds_df = get_plotting_data(
    ps_df=filtered_ps_df,
    ds_df=filtered_ds_df,
    pp_columns=PP_COLUMNS
)

fig = plot_post_strat(
    ps_df=state_ps_df,
    ds_df=state_ds_df,
    samples=n_samples
)
st.header("H. W. Bush Support Before 1988 Elections")
st.pyplot(fig)
