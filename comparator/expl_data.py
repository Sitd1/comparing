import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

import plotly.express as px
from plotly.subplots import make_subplots

from typing import Union
from scipy import stats
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
                 name: str = 'name',
                 feature_descriptions: dict = None,
                 bootstrap_n: int = 10000,
                 do_tqdm=False
                 ):

        self.data = data
        self.name = name
        self.feature_descriptions = feature_descriptions
        self.bootstrap_n = bootstrap_n
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
                        feature: str,
                        opacity: float = 0.3,
                        title: str = None):

        if self.data1.feature_descriptions is not None:
            description = self.data1.feature_descriptions.get(feature, feature)

        if title is None:
            title = f'Сравнение распределений {description}'

        fig = px.histogram(data_frame=self.data1,
                           x=feature,
                           color_discrete_sequence=['blue'],
                           opacity=opacity,
                           labels={feature: self.data1.name}
                           )

        fig.add_trace(
            px.histogram(
                data_frame=self.data2,
                x=feature,
                color_discrete_sequence=['red'],
                opacity=opacity,
                labels={feature: self.data2.name}
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

    def _get_sns_histplot_overlay(self, data1, data2, ax,
                                  plot_name='Title'):
        sns.histplot(data1, kde=True, color="blue",
                     label=self.data1.name, alpha=0.5, ax=ax)
        sns.histplot(data2, kde=True, color="red",
                     label=self.data2.name, alpha=0.5, ax=ax)
        ax.set_title(plot_name)
        ax.legend()

    def get_compare_hists(self, feature):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        if self.data1.feature_descriptions is not None:
            description = self.data1.feature_descriptions.get(feature, feature)
        else:
            description = feature
        # make hists
        properties = ['data', 'booted_data_mean', 'booted_data_median']
        captions = ['Исходные данные',
                    'Бутстрапированные данные (среднее)',
                    'Бутстрапированные данные (медиана)'
        ]

        for i, prop in enumerate(properties):

            data1 = self.data1.__getattribute__(prop)[feature]
            data2 = self.data2.__getattribute__(prop)[feature]
            # Тест на стат различие выборок
            p_value, test_name = make_test_bw_samples(
                data1.values,
                data2.values,
                alpha=self.alpha,
                show_test_name=True
            )
            p_value = round(p_value, 4)
            # Тест Шпиро-Уилка
            test_shapiro = map(stats.shapiro, [data1.values, data2.values])
            ps1, ps2 = map(lambda x: round(x[1], 4), test_shapiro)
            # Оформление заголовка
            title = '\n'.join(
                [captions[i],
                 description,
                 f'Рез.теста Шапиро-Уилка: data1: {ps1}, data2: {ps2}',
                 f'Название теста:{test_name}',
                 f'p_value = {p_value}',
                ]
            )
            # Вывод графика
            self._get_sns_histplot_overlay(
                data1=data1,
                data2=data2,
                ax=axes[i],
                plot_name=title
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
