import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


def plot_post_strat(ps_df, ds_df, samples=100):
    """
    """
    def sns_styleset():
        sns.set(context='paper', style='ticks', font='DejaVu Sans')
        matplotlib.rcParams['figure.dpi']        = 300
        matplotlib.rcParams['axes.linewidth']    = 1
        matplotlib.rcParams['xtick.major.width'] = 1
        matplotlib.rcParams['ytick.major.width'] = 1
        matplotlib.rcParams['xtick.major.size']  = 3
        matplotlib.rcParams['ytick.major.size']  = 3
        matplotlib.rcParams['xtick.minor.size']  = 2
        matplotlib.rcParams['ytick.minor.size']  = 2
        matplotlib.rcParams['font.size']         = 13
        matplotlib.rcParams['axes.titlesize']    = 13
        matplotlib.rcParams['axes.labelsize']    = 13
        matplotlib.rcParams['legend.fontsize']   = 13
        matplotlib.rcParams['xtick.labelsize']   = 13
        matplotlib.rcParams['ytick.labelsize']   = 13

    sns_styleset()
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(12, 10),
        sharey=True
    )
    axs = axs.flatten()

    axs[0].scatter(
        ds_df['support'].values,
        ds_df['state'].values,
        c=ds_df['support'].values,
        cmap='RdBu_r',
        vmin=0,
        vmax=1,
        marker='D',
        edgecolors='k',
        s=40,
        zorder=10
    )
    axs[0].set_title('Disaggregation')

    axs[1].scatter(
        ds_df['support'].values - ps_df['mean'].values,
        ps_df['state'].values,
        c=ds_df['support'].values - ps_df['mean'].values,
        cmap='coolwarm_r',
        vmin=-1,
        vmax=1,
        marker='D',
        edgecolors='k',
        s=40,
        zorder=10
    )
    axs[1].set_title('Change in Estimate')

    axs[2].scatter(
        ps_df['mean'].values,
        ps_df['state'].values,
        c=ps_df['mean'].values,
        cmap='RdBu_r',
        vmin=0,
        vmax=1,
        marker='D',
        edgecolors='k',
        s=40,
        zorder=10
    )
    axs[2].set_title('MultiLevel Regression & PostStratification')

    random_samples = np.random.choice(
        [i for i in range(2000)],
        samples
    )
    for sample in random_samples:

        axs[1].scatter(
            ds_df['support'].values - ps_df[f'support_sample_{sample}'].values,
            ps_df['state'].values,
            c=ds_df['support'].values - ps_df[f'support_sample_{sample}'].values,
            cmap='coolwarm_r',
            vmin=-1,
            vmax=1,
            s=0.5,
            zorder=0
        )

        axs[2].scatter(
            ps_df[f'support_sample_{sample}'].values,
            ps_df['state'].values,
            c=ps_df[f'support_sample_{sample}'].values,
            cmap='RdBu_r',
            vmin=0,
            vmax=1,
            s=0.5,
            zorder=0
        )

    axs[0].set_yticks([i for i in range(1, 52)])
    axs[0].set_ylabel('State')

    for idx, ax in enumerate(axs):

        if idx == 1:
            ax.axvline(
                0.0,
                linestyle=':',
                color='k'
            )
            ax.set_xlim(-1.1, 1.1)
            ax.set_xlabel('Estimated Support Shift')
        else:
            ax.axvline(
                0.5,
                linestyle=':',
                color='k'
            )
            ax.set_xlim(-0.1, 1.1)
            ax.set_xlabel('Estimated Support')

    plt.tight_layout()
    return fig

@st.cache
def get_data(ps_path, ds_path):
    """
    """
    ps_df = pd.read_csv(ps_path)
    ds_df = pd.read_csv(ds_path)
    return ps_df, ds_df

@st.cache
def get_plotting_data(ps_df, ds_df, pp_columns):
    """
    """
    state_ps_df = ps_df.groupby('state')[
        ['N'] + pp_columns
    ].sum().reset_index()
    state_ps_df[pp_columns] = state_ps_df[pp_columns].div(
        state_ps_df['N'],
        axis=0
    )

    state_ps_df['mean'] = state_ps_df[
        pp_columns
    ].mean(1)

    state_ps_df['std'] = state_ps_df[
        pp_columns
    ].std(1)

    state_ds_df = ds_df.groupby('state')[
        ['sum', 'size']
    ].sum().reset_index()
    state_ds_df['support'] = state_ds_df['sum'] / state_ds_df['size']

    state_ds_df = state_ds_df[['state', 'support']]

    for missing_state in list(
        set(state_ps_df['state'].unique()) - set(state_ds_df['state'].unique())
        ):

        state_ds_df = state_ds_df.append(
            {
                'state': missing_state,
                'support': np.nan
            },
            ignore_index=True
        )

    state_ds_df = state_ds_df.sort_values('state')
    state_ps_df = state_ps_df.sort_values('state')

    return state_ps_df, state_ds_df
