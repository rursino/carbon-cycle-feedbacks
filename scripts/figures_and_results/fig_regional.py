""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
sns.set_style('darkgrid')

from core import inv_flux as invf
from core import trendy_flux as TRENDYf

import fig_input_data as id

import pickle

from importlib import reload
reload(invf);
reload(TRENDYf);
reload(id);


""" FIGURES """
def inv_year_regional_cwt(save=False, stat_values=False):
    timeres = 'year'
    invf_dict = {
        "year": id.year_invf
    }

    variables = [['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land'],
                 ['Earth_Ocean', 'South_Ocean', 'Tropical_Ocean', 'North_Ocean']]
    all_subplots = [['42' + str(num) for num in range(i,i+8,2)] for i in range(1,3)]
    colors = ['black', 'blue', 'orange', 'green']

    zip_list = zip(all_subplots, variables)

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    cwt_vals = {}

    for subplots, vars in zip_list:
        ymin, ymax = [], []
        for subplot, var, color in zip(subplots, vars, colors):
            ax[subplot] = fig.add_subplot(subplot)

            index, vals = [], []
            for model in invf_dict[timeres]:
                df = invf_dict[timeres][model].cascading_window_trend(variable = var, window_size=10)
                df = df.sort_index()
                index.append(df.index)
                vals.append(df.values.squeeze())

            pad_max_length = 12 if timeres == "month" else 6
            dataframe = {}
            for ind, val in zip(index, vals):
                # if timeres == "month":
                #     val = bandpass_filter(val, 'low', 365/667)
                for i, v in zip(ind, val):
                    if i in dataframe.keys():
                        dataframe[i].append(v)
                    else:
                        dataframe[i] = [v]
            for ind in dataframe:
                row = np.array(dataframe[ind])
                dataframe[ind] = np.pad(row, (0, pad_max_length - len(row)), 'constant', constant_values=np.nan)

            df = pd.DataFrame(dataframe).T.sort_index()
            df *= 1e3  # if want MtC yr-1 uptake

            x = df.index
            y = df.mean(axis=1)
            std = df.std(axis=1)

            cwt_vals[var] = (pd
                                .DataFrame({'Year': x, 'CWT (MtC/yr/GtC)': y, 'std': std})
                                .set_index('Year')
                            )

            ax[subplot].plot(x, y, color=color)
            ax[subplot].fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)
            ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=1)

            stat_vals[var] = stats.linregress(x, y)

            ymin.append((y - 2*std).min())
            ymax.append((y + 2*std).max())

        delta = 0.1
        for subplot in subplots:
            ax[subplot].set_ylim([min(ymin) - delta * abs(min(ymin)),
                                  max(ymax) + delta * abs(max(ymax))])

    ax["421"].set_xlabel("Land", fontsize=14, labelpad=5)
    ax["421"].xaxis.set_label_position('top')
    ax["422"].set_xlabel("Ocean", fontsize=14, labelpad=5)
    ax["422"].xaxis.set_label_position('top')
    for subplot, region in zip(["421", "423", "425", "427"],
                               ["Earth", "South", "Tropical", "North"]):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$  " + " (MtC.yr$^{-1}$.GtC$^{-1}$)",
                    fontsize=16,
                    labelpad=35
                  )

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"inv_regional_{timeres}_cwt.png")

    if stat_values:
        return stat_vals

    return cwt_vals

def inv_seasonal_regional_cwt(save=False, stat_values=False):
    invf_dict = {
        "summer": id.summer_invf,
        "winter": id.winter_invf
    }

    timeres = ['winter', 'summer']
    variables = ['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land']
    all_subplots = [['42' + str(num) for num in range(i,i+8,2)] for i in range(1,3)]

    colors = ['black', 'blue', 'orange', 'green']

    zip_list = zip(all_subplots, timeres)

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    cwt_vals = {}

    for subplots, time in zip_list:
        ymin, ymax = [], []
        for subplot, var, color in zip(subplots, variables, colors):
            ax[subplot] = fig.add_subplot(subplot)

            index, vals = [], []
            for model in invf_dict[time]:
                df = invf_dict[time][model].cascading_window_trend(variable = var, window_size=10)
                df = df.sort_index()
                index.append(df.index)
                vals.append(df.values.squeeze())

            pad_max_length = 6
            dataframe = {}
            for ind, val in zip(index, vals):
                for i, v in zip(ind, val):
                    if i in dataframe.keys():
                        dataframe[i].append(v)
                    else:
                        dataframe[i] = [v]
            for ind in dataframe:
                row = np.array(dataframe[ind])
                dataframe[ind] = np.pad(row, (0, pad_max_length - len(row)), 'constant', constant_values=np.nan)

            df = pd.DataFrame(dataframe).T.sort_index()
            df *= 1e3  # if want MtC yr-1 uptake

            x = df.index
            y = df.mean(axis=1)
            std = df.std(axis=1)

            cwt_vals[var] = (pd
                                .DataFrame({'Year': x, 'CWT (MtC/yr/GtC)': y, 'std': std})
                                .set_index('Year')
                            )

            ax[subplot].plot(x, y, color=color)
            ax[subplot].fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)
            ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=1)

            stat_vals[var] = stats.linregress(x, y)

            ymin.append((y - 2*std).min())
            ymax.append((y + 2*std).max())

        delta = 0.1
        for subplot in subplots:
            ax[subplot].set_ylim([min(ymin) - delta * abs(min(ymin)),
                                  max(ymax) + delta * abs(max(ymax))])

    ax["421"].set_xlabel("Winter", fontsize=14, labelpad=5)
    ax["421"].xaxis.set_label_position('top')
    ax["422"].set_xlabel("Summer", fontsize=14, labelpad=5)
    ax["422"].xaxis.set_label_position('top')
    for subplot, region in zip(["421", "423", "425", "427"],
                               ["Earth", "South", "Tropical", "North"]):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$  " + " (MtC.yr$^{-1}$.GtC$^{-1}$)",
                    fontsize=16,
                    labelpad=35
                  )

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"inv_regional_seasonal_cwt.png")

    if stat_values:
        return stat_vals

    return cwt_vals

def trendy_regional_cwt(timeres, save=False, stat_values=False):
    trendy_dict = {
        "year": (id.year_S1_trendy,
                 id.year_S3_trendy
                ),
        "winter": (id.winter_S1_trendy,
                 id.winter_S3_trendy
                ),
        "summer": (id.summer_S1_trendy,
                 id.summer_S3_trendy
                ),
    }

    vars = ['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land']
    all_subplots = [['42' + str(num) for num in range(i,i+8,2)] for i in range(1,3)]
    colors = ['black', 'blue', 'orange', 'green']

    zip_list = zip(all_subplots, ['S1', 'S3'], trendy_dict[timeres])

    stat_vals = []
    fig = plt.figure(figsize=(16,8))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    cwt_vals = {}

    for subplots, sim, trendy_sim in zip_list:
        ymin, ymax = [], []
        stat_vals_sim = {}
        cwt_sim = {}
        for subplot, var, color in zip(subplots, vars, colors):
            ax[subplot] = fig.add_subplot(subplot)

            index, vals = [], []
            for model in trendy_sim:
                df = trendy_sim[model].cascading_window_trend(variable = var, window_size=10)
                index.append(df.index)
                vals.append(df.values.squeeze())

            pad_max_length = 5
            dataframe = {}
            for ind, val in zip(index, vals):
                for i, v in zip(ind, val):
                    if i in dataframe.keys():
                        dataframe[i].append(v)
                    else:
                        dataframe[i] = [v]
            for ind in dataframe:
                row = np.array(dataframe[ind])
                dataframe[ind] = np.pad(row, (0, pad_max_length - len(row)), 'constant', constant_values=np.nan)

            df = pd.DataFrame(dataframe).T.sort_index()
            df *= 1e3  # if want MtC yr-1 uptake

            x = df.index
            y = df.mean(axis=1)
            std = df.std(axis=1)

            cwt_sim[var] = (pd
                                .DataFrame({'Year': x, 'CWT (MtC/yr/GtC)': y, 'std': std})
                                .set_index('Year')
                            )

            ax[subplot].plot(x, y, color=color)
            ax[subplot].fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)
            ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=1)

            stat_vals_sim[var] = stats.linregress(x, y)

            ymin.append((y - 2*std).min())
            ymax.append((y + 2*std).max())

        stat_vals.append(stat_vals_sim)
        cwt_vals[sim] = cwt_sim

        delta = 0.1
        for subplot in subplots:
            ax[subplot].set_ylim([min(ymin) - delta * abs(min(ymin)),
                                  max(ymax) + delta * abs(max(ymax))])

    ax["421"].set_xlabel("S1", fontsize=14, labelpad=5)
    ax["421"].xaxis.set_label_position('top')
    ax["422"].set_xlabel("S3", fontsize=14, labelpad=5)
    ax["422"].xaxis.set_label_position('top')
    for subplot, region in zip(["421", "423", "425", "427"],
                               ["Earth", "South", "Tropical", "North"]):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$  " + " (MtC.yr$^{-1}$.GtC$^{-1}$)",
                    fontsize=16,
                    labelpad=35
                  )

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"trendy_regional_{timeres}_cwt.png")

    if stat_values:
        return stat_vals

    return cwt_vals

def trendy_regional_cwt_diff(timeres='year', save=False, stat_values=False):
    trendy_dict = {
        "year": (id.year_S1_trendy,
                 id.year_S3_trendy
                ),
        "summer": (id.summer_S1_trendy,
                   id.summer_S3_trendy
                ),
        "winter": (id.winter_S1_trendy,
                   id.winter_S3_trendy
                ),
    }

    variables = ['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land']
    subplots = ['411', '412', '413', '414']
    colors = ['black', 'blue', 'orange', 'green']

    zip_list = zip(subplots, variables, colors)

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    cwt_vals = {}

    ymin, ymax = [], []
    for subplot, var, color in zip_list:
        ax[subplot] = fig.add_subplot(subplot)

        all_index, all_vals = [], []
        for sim in trendy_dict[timeres]:
            sim_index, sim_vals = [], []
            for model in sim:
                model_df = sim[model].cascading_window_trend(variable = var, window_size=10)
                sim_index.append(model_df.index)
                sim_vals.append(model_df.values.squeeze())
            all_index.append(sim_index)
            all_vals.append(sim_vals)

        index = all_index[0]
        vals = []
        for i in range(len(all_vals[0])):
            vals.append(np.array(all_vals[1][i]) - np.array(all_vals[0][i]))

        pad_max_length = 5
        dataframe = {}
        for ind, val in zip(index, vals):
            for i, v in zip(ind, val):
                if i in dataframe.keys():
                    dataframe[i].append(v)
                else:
                    dataframe[i] = [v]
        for ind in dataframe:
            row = np.array(dataframe[ind])
            dataframe[ind] = np.pad(row, (0, pad_max_length - len(row)), 'constant', constant_values=np.nan)

        df = pd.DataFrame(dataframe).T.sort_index().iloc[3:]
        df *= 1e3  # if want MtC yr-1 uptake

        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        cwt_vals[var] = (pd
                            .DataFrame({'Year': x, 'CWT (MtC/yr/GtC)': y, 'std': std})
                            .set_index('Year')
                        )

        ax[subplot].plot(x, y, color=color)
        ax[subplot].fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)
        ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=1)

        stat_vals[var] = stats.linregress(x, y)

        ymin.append((y - 2*std).min())
        ymax.append((y + 2*std).max())

    for subplot, region in zip(subplots,
                               ["Earth", "South", "Tropical", "North"]):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

        delta = 0.1
        ax[subplot].set_ylim([min(ymin) - delta * abs(min(ymin)),
                              max(ymax) + delta * abs(max(ymax))])

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$  " + " (MtC.yr$^{-1}$.GtC$^{-1}$)",
                    fontsize=16,
                    labelpad=35
                  )

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"trendy_regional_{timeres}_cwt_diff.png")

    if stat_values:
        return stat_vals

    return cwt_vals


""" EXECUTION """
inv_year_regional_cwt(save=False)
inv_seasonal_regional_cwt(save=False)

trendy_regional_cwt('year', save=False)
trendy_regional_cwt('winter', save=False)
trendy_regional_cwt('summer', save=False)

trendy_regional_cwt_diff('year', save=False)
trendy_regional_cwt_diff('winter', save=False)
trendy_regional_cwt_diff('summer', save=False)


# PICKLE (YET TO BE UPDATED)
pickle.dump(inv_regional_cwt("year"), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_regional_year_cwt.pik', 'wb'))
pickle.dump(inv_regional_cwt("month"), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_regional_month_cwt.pik', 'wb'))

pickle.dump(trendy_regional_cwt('year'), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_regional_year_cwt.pik', 'wb'))
pickle.dump(trendy_regional_cwt('month'), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_regional_month_cwt.pik', 'wb'))

pickle.dump(trendy_regional_cwt_diff('year'), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_regional_year_cwt_diff.pik', 'wb'))
pickle.dump(trendy_regional_cwt_diff('month'), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_regional_month_cwt_diff.pik', 'wb'))
