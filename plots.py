import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, List, Tuple
import numpy as np
from pandas import Timestamp

def plot_normalized_prices(price_data: pd.DataFrame, ticker_map: Dict[str, str]) -> Optional[go.Figure]:
    """
    Строит график нормализованных цен активов.

    Args:
        price_data (pd.DataFrame): DataFrame с ценами закрытия активов (индекс - дата).
        ticker_map (Dict[str, str]): Словарь для отображения тикеров как названий.

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None, если входные данные некорректны.
    """
    if price_data is None or price_data.empty:
        return None

    price_data_assets = price_data.drop(columns=['Cash'], errors='ignore')
    if price_data_assets.empty:
        print("Warning: Price data for normalization is empty.")
        return None

    try:
        if (price_data_assets.iloc[0] == 0).any():
            print("Warning: Zero prices found in the first row. Cannot normalize.")
            return None
        normalized_data = price_data_assets / price_data_assets.iloc[0] * 100
    except Exception as e:
        print(f"Error during price normalization: {e}")
        return None

    fig = go.Figure()

    for ticker in normalized_data.columns:
        display_name = ticker_map.get(ticker, ticker)
        fig.add_trace(go.Scatter(
            x=normalized_data.index,
            y=normalized_data[ticker],
            mode='lines',
            name=display_name,
        ))

    fig.update_layout(
        # title='Динамика нормализованных цен активов (база 100 на начало периода)', # Убрано
        xaxis_title='Дата',
        yaxis_title='Нормализованная цена',
        legend_title='Активы',
        legend=dict(font=dict(size=10)),
        hovermode='x unified'
    )
    return fig

def plot_equity_curves(backtest_results: pd.DataFrame, ticker_map: Dict[str, str],
                         hover_weights_text: pd.DataFrame) -> Optional[go.Figure]:
    """
    Строит график кривых эквити для всех рассчитанных стратегий.

    Args:
        backtest_results (pd.DataFrame): DataFrame с результатами бэктеста.
        ticker_map (Dict[str, str]): Словарь для отображения тикеров как названий.
        hover_weights_text (pd.DataFrame): DataFrame с текстом весов для hover.

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None.
    """
    if backtest_results is None or backtest_results.empty:
        return None

    strategy_columns = {}
    if 'Calendar_Rebalanced_Value' in backtest_results.columns:
        strategy_columns['Calendar_Rebalanced_Value'] = 'Ребаланс (Календарь)'
    if 'Weight_Band_Value' in backtest_results.columns:
        strategy_columns['Weight_Band_Value'] = 'Ребаланс (Доля %)'
    if 'Combined_Value' in backtest_results.columns:
        strategy_columns['Combined_Value'] = 'Ребаланс (Комби)'
    if 'BH_Target_Value' in backtest_results.columns:
        strategy_columns['BH_Target_Value'] = 'B&H (Целевые веса)'
    for col in backtest_results.columns:
        if col.startswith('BH_') and col != 'BH_Target_Value':
            ticker = col.split('_', 1)[1]
            display_ticker_name = ticker_map.get(ticker, ticker)
            strategy_columns[col] = f'B&H ({display_ticker_name})'
    main_strategies = ['Calendar_Rebalanced_Value', 'Weight_Band_Value', 'Combined_Value', 'BH_Target_Value']

    fig = go.Figure()

    # --- Обновляем шаблоны Hover --- 
    hover_template_rebalance = '<b>%{fullData.name}: %{y:$,.0f}</b><br>Доли: %{text}<extra></extra>' # Убрали дату, перенесли стоимость
    hover_template_bh = '<b>%{fullData.name}: %{y:$,.0f}</b><extra></extra>' # Убрали дату, перенесли стоимость
    # --------------------------------

    # --- Отрисовка кривых эквити с hover --- 
    for col_name in main_strategies:
        if col_name in backtest_results.columns:
            display_name = strategy_columns.get(col_name, col_name)
            text_series = None
            template = hover_template_bh
            if col_name != 'BH_Target_Value' and col_name in hover_weights_text.columns:
                text_series = hover_weights_text[col_name]
                template = hover_template_rebalance

            fig.add_trace(go.Scatter(
                x=backtest_results.index,
                y=backtest_results[col_name],
                mode='lines',
                name=display_name,
                text=text_series,
                hovertemplate=template # Используем обновленный шаблон
            ))

    # Для индивидуальных B&H используем обновленный шаблон без долей
    for col_name, display_name in strategy_columns.items():
        if col_name not in main_strategies and col_name in backtest_results.columns:
             fig.add_trace(go.Scatter(
                 x=backtest_results.index,
                 y=backtest_results[col_name],
                 mode='lines',
                 name=display_name,
                 line=dict(dash='dot'),
                 hovertemplate=hover_template_bh # Обновленный шаблон без долей
             ))
    # -----------------------------------------

    fig.update_layout(
        # title='Динамика стоимости портфелей', # Убрано
        xaxis_title='Дата',
        yaxis_title='Стоимость портфеля',
        yaxis_type='log',
        legend_title='Стратегия',
        legend=dict(font=dict(size=10)),
        hovermode='x unified'
    )
    return fig

def plot_drawdown_curves(drawdown_results: pd.DataFrame) -> Optional[go.Figure]:
    """Строит график временных рядов просадок для основных стратегий.

    Args:
        drawdown_results (pd.DataFrame): DataFrame с рядами просадок.

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None.
    """
    if drawdown_results is None or drawdown_results.empty:
        return None

    fig = go.Figure()

    # Определяем имена для легенды (совпадают с именами в calculate_metrics)
    strategy_names_map = {
        'Calendar_Rebalanced_Value': 'Ребаланс (Календарь)',
        'Weight_Band_Value': 'Ребаланс (Доля %)',
        'Combined_Value': 'Ребаланс (Комби)',
        'BH_Target_Value': 'B&H (Целевые веса)'
    }

    for col in drawdown_results.columns:
        if col in strategy_names_map:
            # Умножаем на 100 для отображения в процентах
            drawdown_percent = drawdown_results[col] * 100
            display_name = strategy_names_map[col]
            fig.add_trace(go.Scatter(
                x=drawdown_percent.index,
                y=drawdown_percent,
                mode='lines',
                name=display_name
            ))

    fig.update_layout(
        # title='Динамика просадок портфелей', # Убрано
        xaxis_title='Дата',
        yaxis_title='Просадка (%)',
        legend_title='Стратегия',
        legend=dict(font=dict(size=10)),
        hovermode='x unified',
        yaxis_tickformat='.1f' # Формат оси Y с одним знаком после запятой
    )
    # Устанавливаем диапазон оси Y, чтобы 0 был наверху
    min_drawdown = drawdown_results.min().min() * 100 # Минимальная просадка по всем стратегиям в %
    if not pd.isna(min_drawdown):
         fig.update_yaxes(range=[min_drawdown * 1.1, 5]) # Чуть ниже минимума и до 5%
    else:
         fig.update_yaxes(range=[-50, 5]) # Диапазон по умолчанию, если нет данных

    return fig

def plot_correlation_heatmap(price_data: pd.DataFrame, ticker_map: Dict[str, str]) -> Optional[go.Figure]:
    """Строит тепловую карту корреляций дневных доходностей активов.

    Args:
        price_data (pd.DataFrame): DataFrame с ценами активов (индекс - дата).
        ticker_map (Dict[str, str]): Словарь для отображения тикеров как названий.

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None.
    """
    if price_data is None or price_data.empty:
        return None

    # Оставляем только активы, исключая Кэш
    asset_prices = price_data.drop(columns=['Cash'], errors='ignore')

    if asset_prices.shape[1] < 2:
        print("Warning: Need at least 2 assets to calculate correlation.")
        return None

    # Расчет дневных доходностей
    daily_returns = asset_prices.pct_change().dropna()

    if daily_returns.empty:
        print("Warning: Empty daily returns, cannot calculate correlation.")
        return None

    # Расчет матрицы корреляций
    corr_matrix = daily_returns.corr()

    # Создание маски для верхнего треугольника (включая диагональ)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Применение маски (замена верхнего треуг. и диагонали на NaN)
    corr_matrix_masked = corr_matrix.where(~mask)

    # Замена тикеров на полные названия для осей
    axis_labels = [ticker_map.get(t, t) for t in corr_matrix.columns]

    # Создание матрицы текста: округленные значения, пустые строки для маски
    text_labels = corr_matrix.round(2).astype(str)
    text_labels[mask] = "" # Заменяем ненужные значения на пустые строки

    fig = go.Figure(data=go.Heatmap(
           z=corr_matrix_masked, # Маскированная матрица для цвета
           x=axis_labels, # Используем названия для осей
           y=axis_labels,
           text=text_labels, # Матрица с пустыми строками для текста
           texttemplate="%{text}", # Отображаем текст как есть (число или пусто)
           textfont={"size":10},
           hoverongaps = False,
           colorscale='RdBu', # Шкала от красного (-) к синему (+)
           zmid=0, # Центрируем шкалу на 0
           zmin=-1, # Устанавливаем мин/макс для консистентности шкалы
           zmax=1
    ))

    fig.update_layout(
        # title='Матрица корреляций дневных доходностей активов', # Убрано
        xaxis_title='',
        yaxis_title='',
        xaxis_tickangle=-45, # Поворот подписей по X
        yaxis_autorange='reversed', # Переворачиваем ось Y для стандартного вида матрицы
        width=700, # Фиксированная ширина для читаемости
        height=600, # Фиксированная высота
        margin=dict(l=150, r=50, t=50, b=150) # Отступы для длинных названий
    )
    fig.update_xaxes(side="bottom") # Перемещаем X ось вниз

    return fig

def plot_rolling_volatility(volatility_df: pd.DataFrame, ticker_map: Dict[str, str]) -> Optional[go.Figure]:
    """График скользящей волатильности (Inverse Volatility Input)."""
    if volatility_df is None or volatility_df.empty:
        return None
        
    fig = go.Figure()
    
    # Приводим к годовой волатильности для отображения?
    # Обычно волатильность показывают в % годовых
    annualized_vol = volatility_df * np.sqrt(252) * 100
    
    for ticker in annualized_vol.columns:
        display_name = ticker_map.get(ticker, ticker)
        fig.add_trace(go.Scatter(
            x=annualized_vol.index,
            y=annualized_vol[ticker],
            mode='lines',
            name=display_name,
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
    fig.update_layout(
        xaxis_title='Дата',
        yaxis_title='Волатильность (% годовых)',
        legend_title='Активы',
        hovermode='x unified'
    )
    return fig

def plot_weighted_risk_contribution(
    volatility_df: pd.DataFrame, 
    weights_df: pd.DataFrame,
    ticker_map: Dict[str, str]
) -> Optional[go.Figure]:
    """График взвешенного риска (Weight * Volatility)."""
    if volatility_df is None or weights_df is None:
        return None
        
    # Выравниваем индексы
    common_idx = volatility_df.index.intersection(weights_df.index)
    vol_aligned = volatility_df.loc[common_idx]
    w_aligned = weights_df.loc[common_idx]
    
    # Умножаем вес на волатильность (вклад в риск)
    # Volatility в долях, weights в долях.
    # Результат * 100 для %
    risk_contribution = vol_aligned * w_aligned * np.sqrt(252) * 100
    
    fig = go.Figure()
    
    for ticker in risk_contribution.columns:
        if ticker == 'Cash': continue # Пропускаем Cash
        display_name = ticker_map.get(ticker, ticker)
        fig.add_trace(go.Scatter(
            x=risk_contribution.index,
            y=risk_contribution[ticker],
            mode='lines',
            name=display_name,
             hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
    fig.update_layout(
        xaxis_title='Дата',
        yaxis_title='Вклад в риск (% Vol * Weight)',
        legend_title='Активы',
        hovermode='x unified',
        title="Равномерность распределения риска (Ideal: линии должны сходиться)"
    )
    return fig

def plot_effective_weights(weights_df: pd.DataFrame, ticker_map: Dict[str, str], rebalance_dates: Optional[List[pd.Timestamp]] = None) -> Optional[go.Figure]:
    """График динамики целевых весов (Stacked Area)."""
    if weights_df is None or weights_df.empty:
        return None
    
    fig = go.Figure()
    
    # Сортируем колонки, чтобы Cash был последним или первым для красоты?
    # Обычно кэш вниз или вверх. Оставим как есть.
    
    for col in weights_df.columns:
        display_name = ticker_map.get(col, col)
        fig.add_trace(go.Scatter(
            x=weights_df.index,
            y=weights_df[col] * 100, # В процентах
            mode='lines',
            name=display_name,
            stackgroup='one', # Включаем стекинг
            groupnorm='percent', # Нормализация до 100% (для гарантии)
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
    
    # Add vertical lines for rebalancing events
    if rebalance_dates is not None and len(rebalance_dates) > 0:
        shapes = []
        for d in rebalance_dates:
            shapes.append(dict(
                type="line",
                x0=d,
                y0=0,
                x1=d,
                y1=100,
                xref="x",
                yref="y",
                layer="above",
                line=dict(
                    color="grey",
                    width=1,
                    dash="solid"
                )
            ))
        fig.update_layout(shapes=shapes)
    
    fig.update_layout(
        xaxis_title='Дата',
        yaxis_title='Целевой Вес (%)',
        legend_title='Активы',
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    return fig