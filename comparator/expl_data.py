import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

import plotly.express as px
from plotly.subplots import make_subplots

from typing import Union
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .bootstrap import get_bootstraped_data, tqdm_

# FixMe
pd.options.plotting.backend = "plotly"

config_plotly = {'modeBarButtonsToAdd': [
    'drawopenpath', 'hoverCompare'
]}
sw = dict(config=config_plotly,  renderer='iframe')


class DataToCompare:
    def __init__(self,
                 data: pd.DataFrame,
                 label_column_name: str,
                 feature_descriptions: dict = None,
                 label_values: list = None,
                 color_mapping: dict = None,
                 bootstrap_n: int = 10000,
                 do_tqdm=False
                 ):

        self.data = data
        self.label_column_name = label_column_name
        self.feature_descriptions = feature_descriptions
        self.bootstrap_n = bootstrap_n
        self.label_values = label_values
        self.color_mapping = color_mapping
        self.do_tqdm = do_tqdm

        self._booted_data_mean = None
        self._booted_data_median = None
        self._smote_oversampled_data = None
        self._shapiro = None

    def make_booted_data(self, n, statistic='median', do_tqdm=False):
        booted_data = {}
        for column in self.data.columns:
            booted_data[column] = get_bootstraped_data(
                control_data=self.data[column],
                n=self.bootstrap_n,
                statistic=statistic,
                do_tqdm=do_tqdm,
            )
        return pd.DataFrame(booted_data)

    @property
    def booted_data_mean(self):
        if self._booted_data_mean is None:
            self.set_booted_data_mean()
        return self._booted_data_mean

    def set_booted_data_mean(self, n=None, do_tqdm=False):
        if n is None:
            n = self.bootstrap_n
        self._booted_data_mean = self.make_booted_data(n=n, statistic='mean',do_tqdm=do_tqdm)

    @property
    def booted_data_median(self):
        if self._booted_data_median is None:
            self.set_booted_data_median()
        return self._booted_data_median

    def set_booted_data_median(self, n=None, do_tqdm=False):
        if n is None:
            n = self.bootstrap_n
        self._booted_data_median = self.make_booted_data(n=n, statistic='median', do_tqdm=do_tqdm)

    @property
    def shapiro(self):
        if self._shapiro is None:

            shapiro = dict()
            for prop in ['data', 'booted_data_mean', 'booted_data_median']:
                local_value = dict()
                df = self.__getattribute__(prop)
                for column in df.columns:
                    _, local_value[column] = stats.shapiro(df[column].values)
                shapiro[prop] = local_value

            self._shapiro = pd.DataFrame(shapiro)
        return self._shapiro

class ComparingData:
    def __init__(self,
                 data1: DataToCompare,
                 data2: DataToCompare,
                 alpha=0.05
                 ):
        self.data1 = data1
        self.data2 = data2
        self.alpha = alpha
        self._p_values = None
        self._smote_oversampled_data = None
        if all(self.data1.data.columns != self.data2.data.columns):
            raise ValueError('Columns data1 should be equal to columns data2')
    # FixMe
    @property
    def tt_p_values(self):
        if self._p_values is None:
            result = dict()
            for prop in ['data', 'booted_data_mean', 'booted_data_median']:
                p_values = dict()
                test_names = dict()
                data1 = self.data1.__getattribute__(prop)
                data2 = self.data2.__getattribute__(prop)
                for column in data1.columns:
                    p_values[column], test_names[column] = make_test_bw_samples(
                        data1=data1[column],
                        data2=data2[column],
                        alpha=0.05,
                        show_test_name=True
                    )
                result[f'{prop}_test_name'] = test_names
                result[f'{prop}_p_value'] = p_values
            self._p_values = pd.DataFrame(result)
        return self._p_values

    @property
    def smote_oversampled_data(self):
        if self._smote_oversampled_data is None:
            pass
            # smote = SMOTE(sampling_strategy='auto')
            # X_resampled, y_resampled = smote.fit_resample(X, y)
        return self._smote_oversampled_data

    def get_plotly_hist(self,
                        column_name: str,
                        opacity: float = 0.3,
                        title: str = None):

        if self.data1.feature_descriptions is not None:
            description = self.data1.feature_descriptions.get(column_name, column_name)

        if title is None:
            title = f'Сравнение распределений "{description}"'

        fig = px.histogram(data_frame=self.data1.data,
                           x=column_name,
                           color_discrete_sequence=['blue'],
                           opacity=opacity,
                           labels={column_name: self.data1.name}
                           )

        fig.add_trace(
            px.histogram(
                data_frame=self.data2.data,
                x=column_name,
                color_discrete_sequence=['red'],
                opacity=opacity,
                labels={column_name: self.data2.name}
            ).data[0]
        )

        fig.update_layout(
            title=title,
            xaxis=dict(title='Значения'),
            yaxis=dict(title='Частота'),
            barmode='overlay'
        )

        fig.update_traces(showlegend=True)
        fig.show(**sw)

    def _get_diff_bootstrap_vals_hist(self,
                                      column_name,
                                      statistic='mean',
                                      bootstrap_conf_level: float = 0.95):
        suffix = f'booted_data_{statistic}'
        # находим разницу средних -> np.array
        first = self.data1.__getattribute__(suffix)[column_name].values
        second = self.data2.__getattribute__(suffix)[column_name].values
        booted_datas_diff = first - second

        # находим доверительный интервал bootstrap_conf_level
        left_quant = (1 - bootstrap_conf_level) / 2
        right_quant = 1 - (1 - bootstrap_conf_level) / 2
        quants = np.quantile(booted_datas_diff, [left_quant, right_quant])

        #
        p_1 = stats.norm.cdf(
            x=0,
            loc=np.mean(booted_datas_diff),
            scale=np.std(booted_datas_diff)
        )
        p_2 = stats.norm.cdf(
            x=0,
            loc=-np.mean(booted_datas_diff),
            scale=np.std(booted_datas_diff)
        )

        p_value = min(p_1, p_2) * 2

        # # визуализация
        # sns.histplot(booted_datas_diff, kde=False, color="blue",
        #              label='gis_avg_permeability', alpha=0.5)

        # Визуализация
        _, _, bars = plt.hist(booted_datas_diff, bins=50)

        for bar in bars:
            if abs(bar.get_x()) <= quants[0] or abs(bar.get_x()) >= quants[1]:
                bar.set_facecolor('red')
            else:
                bar.set_facecolor('gray')
                bar.set_edgecolor('black')

        max_height = max(bar.get_height() for bar in bars)
        ymax_vlines = (max_height * 0.9).round()

        plt.style.use('ggplot')
        plt.vlines(quants, ymin=0, ymax=ymax_vlines, linestyle='--', color='black')
        plt.xlabel('boot_data')
        plt.ylabel('frequency')
        plt.title("Histogram of boot_data")
        plt.show()


    def _get_sns_histplot_overlay(self, data1, data2, ax,
                                  plot_name='Title'):
        sns.histplot(data1, kde=True, color="blue",
                     label=self.data1.name, alpha=0.5, ax=ax)
        sns.histplot(data2, kde=True, color="red",
                     label=self.data2.name, alpha=0.5, ax=ax)
        ax.set_title(plot_name)
        ax.legend()

    def get_compare_hists(self, feature):

        # if self.data1.feature_descriptions is not None:
        #     description = self.data1.feature_descriptions.get(feature, feature)
        # else:
        #     description = feature

        # # make hists
        # properties = ['booted_data_mean', 'booted_data_median']
        # captions = ['Бутстрапированные данные (среднее)',
        #             'Бутстрапированные данные (медиана)'
        # ]
        #

        # вывод исходного графика
        self.get_plotly_hist(feature)
        # FixMe
        # вывод бустрап по средним значениям
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))

        data1 = self.data1.booted_data_mean[feature]
        data2 = self.data2.booted_data_mean[feature]
        title = 'Бутстрапированные данные (среднее)'

        self._get_sns_histplot_overlay(
            data1=data1,
            data2=data2,
            ax=axes[0],
            plot_name=title
        )

        # вывод разницы бустрап по средним значениям
        self._get_diff_bootstrap_vals_hist(
            column_name=feature,
            statistic='mean',
        )

        # вывод бустрап по медиане
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))

        data1 = self.data1.booted_data_median[feature]
        data2 = self.data2.booted_data_median[feature]
        title = 'Бутстрапированные данные (медиана)'

        self._get_sns_histplot_overlay(
            data1=data1,
            data2=data2,
            ax=axes[0],
            plot_name=title
        )

        # вывод разницы бустрап по медиане
        self._get_diff_bootstrap_vals_hist(
            column_name=feature,
            statistic='median',
        )

    def get_all_hists(self):
        for column in self.data1.data.columns:
            self.get_compare_hists(column)


def is_normal_shapiro(
        data1: Union[np.array, pd.Series],
        data2: Union[np.array, pd.Series],
        alpha: float = 0.05) -> bool:
    test_shapiro = map(stats.shapiro, [data1, data2])
    p_values_less_alpha = map(lambda x: x[1] >= alpha, test_shapiro)
    return all(p_values_less_alpha)


def make_test_bw_samples(
        data1: Union[np.array, pd.Series],
        data2: Union[np.array, pd.Series],
        alpha: float = 0.05,
        show_test_name: bool = False
) -> float:
    if is_normal_shapiro(data1=data1, data2=data2, alpha=alpha):
        test = stats.ttest_ind
    else:
        test = stats.mannwhitneyu
    _, p_value = test(data1, data2)

    if show_test_name:
        test_name = test.__name__
        return p_value, test_name
    return p_value


def make_compare_scater(data: pd.DataFrame,
                     label_column_name: str,
                     embedding_: str = 'tsne',
                     n_components: int = 2,
                     perplexity: int = 30,
                     label_values: list = None,
                     color_mapping: dict = None
):
    """

    """
    scatter_xyz = ('x', 'y', 'z')

    if label_values is None:
        label_values = [1]

    data = data[data.index.notna()]

    if label_values is not None:
        data = data[data[label_column_name].isin(label_values)]

    # pipeline
    scaler = StandardScaler()
    # embeding setting
    if embedding_ == 'tsne':
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=300)
        pipeline = Pipeline([('scaler', scaler), ('tsne', tsne)])
        title = f'Embedding type: TSNE, perplexity={perplexity}'
    elif embedding_ == 'pca':
        pca = PCA(n_components=n_components)
        pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
        title = 'Embedding type: PCA'
    else:
        raise ValueError('"embedding_" Value should be in ["tsne", "pca"]')
    # fit_transform
    embedded_data = pipeline.fit_transform(data)
    # preparation
    embedded_data = pd.DataFrame(dict(zip(data.index, embedded_data))).T
    columns = embedded_data.columns
    embedded_data.loc[:, label_column_name] = data.loc[:, label_column_name].astype('str')
    # data_scaled_tsne['g_type_'] = data_scaled_tsne['g_type_'].fillna('other')

    prop = dict(zip(scatter_xyz, columns))
    conf = dict(data_frame=embedded_data.reset_index(),
                **prop,
                color=label_column_name,
                color_discrete_map=color_mapping,
                hover_data=['index', label_column_name],
                color_continuous_scale='Portland',
                opacity=0.7,
                title=title)
    if n_components == 2:
        px.scatter(**conf).show(**sw)
    elif n_components == 3:
        px.scatter_3d(**conf).show(**sw)
    else:
        raise ValueError('n_components should be in [2, 3]')
