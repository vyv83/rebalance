import pandas as pd
import numpy as np
# Прямая загрузка данных через yfinance
import streamlit as st
from datetime import date
from typing import List, Dict, Optional, Tuple, Literal
from pandas import Timestamp

import yfinance as yf

# --- Константы ---
TRADING_DAYS_PER_YEAR = 252

# --- Функции загрузки данных ---

@st.cache_data
def load_price_data(
    tickers: List[str],
    start_date: date,
    end_date: date
) -> Optional[pd.DataFrame]:
    """
    Загружает и обрабатывает исторические данные через yfinance.
    Ожидает на выходе DataFrame с колонкой 'Close' и 'Cash'.
    """
    if not tickers:
        print("Error: No tickers provided.")
        return None

    try:
        # Загружаем данные через yfinance
        print(f"Loading data for {tickers} from {start_date} to {end_date} using yfinance...")
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        # Проверяем, что данные загружены
        if data is None or data.empty:
            print("Error: No data received from yfinance.")
            return None

        # Используем Close цены
        if isinstance(data.columns, pd.MultiIndex):
            # Если MultiIndex (для нескольких тикеров), берем только 'Close'
            close_data = data['Close']
        else:
            # Для одного тикера
            close_data = data[['Close']] if 'Close' in data.columns else data

        # Удаляем строки с NaN
        close_data = close_data.dropna()

        # Проверяем, что остались данные
        if close_data.empty:
            print("Error: All data was removed after dropna.")
            return None

        # Добавляем столбец 'Cash' со значением 1.0
        close_data['Cash'] = 1.0

        print(f"Successfully loaded data with shape: {close_data.shape}")
        return close_data

    except Exception as e:
        print(f"Error loading data from yfinance: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Функции бэктестинга ---

# Функция для расчета просадки
def calculate_drawdown_series(series: pd.Series) -> pd.Series:
    """Рассчитывает временной ряд просадки для заданной серии стоимости."""
    cumulative_max = series.cummax()
    # Избегаем деления на 0, заменяя 0 на NaN
    cumulative_max_safe = cumulative_max.replace(0, np.nan)
    drawdown = (series - cumulative_max_safe) / cumulative_max_safe
    # Заполняем NaN нулями (например, в начале, пока просадки нет)
    drawdown = drawdown.fillna(0)
    return drawdown

# Вспомогательная функция для форматирования весов
def format_hover_weights(holdings_dict: Dict[str, float], cash_value: float,
                         prices: pd.Series, assets_to_format: List[str],
                         all_assets_with_price: List[str]) -> str:
    """Форматирует строку с весами для hover tooltip."""
    asset_values = {}
    total_asset_value = 0.0
    for ticker in all_assets_with_price:
        shares = holdings_dict.get(ticker, 0.0)
        price = prices.get(ticker, np.nan)
        value = 0.0
        if shares > 0 and not pd.isna(price) and price > 0:
            value = shares * price
        asset_values[ticker] = value
        total_asset_value += value

    total_value = total_asset_value + cash_value

    hover_parts = []
    if total_value > 1e-6:
        # Сначала активы, входящие в целевые
        for ticker in assets_to_format:
            weight = asset_values.get(ticker, 0.0) / total_value
            hover_parts.append(f"{ticker}{weight*100:.0f}")
        # Затем кэш
        cash_weight = cash_value / total_value
        hover_parts.append(f"Cash{cash_weight*100:.0f}")
    else:
        return "N/A"

    return '/'.join(hover_parts)

@st.cache_data
def run_backtest(
    price_data: pd.DataFrame,
    target_weights: Dict[str, float],
    rebalance_freq: str,
    initial_capital: float,
    weight_deviation_threshold: float,
    deviation_type: Literal['absolute', 'relative']
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[Tuple[Timestamp, str]]], pd.DataFrame]]:
    """
    Выполняет бэктестинг стратегий ребалансировки и сравнение с Buy & Hold.

    Args:
        price_data (pd.DataFrame): DataFrame с ценами (включая 'Cash').
        target_weights (Dict[str, float]): Словарь целевых весов (доли, сумма = 1.0).
        rebalance_freq (str): Частота ребалансировки (Календарная: 'ME', 'QE', 'YE').
        initial_capital (float): Начальный капитал.
        weight_deviation_threshold (float): Абсолютный порог отклонения доли в % для запуска ребалансировки.
        deviation_type (Literal['absolute', 'relative']): Тип порога отклонения.

    Returns:
        Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[Tuple[Timestamp, str]]], pd.DataFrame]]:
            Кортеж из четырех элементов:
            1. results: DataFrame со стоимостями портфелей для разных стратегий.
            2. drawdown_results: DataFrame с рядами просадок для основных стратегий.
            3. rebalance_log: Словарь с логами ребалансировок {strategy_name: [(date, type)]}.
            4. hover_weights_text: DataFrame с текстом весов для hover.
            Или None в случае ошибки.
    """
    if price_data is None or price_data.empty or not target_weights:
        return None

    assets = [col for col in target_weights.keys() if col != 'Cash']
    if not all(asset in price_data.columns for asset in assets):
         print("Error: Price data missing for some assets defined in target_weights.")
         return None

    # Конвертируем порог из % в долю
    threshold_weight_delta_fraction = weight_deviation_threshold / 100.0

    # Определяем активы для ребалансировки (из target_weights) и все активы (из price_data)
    assets_to_rebalance = [col for col in target_weights.keys() if col != 'Cash']
    all_price_assets = [col for col in price_data.columns if col != 'Cash'] # Все активы с ценами

    # --- Общая инициализация для всех стратегий ребалансировки ---
    first_date = price_data.index[0]

    # Рассчитываем начальные доли и кэш ОДИН РАЗ
    initial_total_value = initial_capital
    initial_holdings_template = {} # Шаблон долей
    for ticker in assets_to_rebalance:
        target_value = initial_total_value * target_weights.get(ticker, 0.0)
        initial_price = price_data.at[first_date, ticker]
        shares = 0.0
        if not pd.isna(initial_price) and initial_price > 0 and not pd.isna(target_value):
            shares = target_value / initial_price
        else:
            print(f"Warning: Initial price/target issue for {ticker}. Setting shares to 0.")
        initial_holdings_template[ticker] = shares
    # Добавляем нулевые доли для активов, не входящих в target_weights
    for ticker in all_price_assets:
         if ticker not in initial_holdings_template:
              initial_holdings_template[ticker] = 0.0
    initial_cash = initial_total_value * target_weights.get('Cash', 0.0)

    # Рассчитываем начальную стоимость портфеля (для проверки)
    initial_asset_value_check = 0.0
    for ticker in all_price_assets:
         shares = initial_holdings_template.get(ticker, 0.0)
         price = price_data.at[first_date, ticker]
         if not pd.isna(price) and price > 0 and shares > 0:
             initial_asset_value_check += shares * price
    initial_total_value_check = initial_asset_value_check + initial_cash
    if not np.isclose(initial_total_value_check, initial_capital):
        print(f"Warning: Initial calculated value check ({initial_total_value_check:.2f}) differs from capital ({initial_capital:.2f}).")
    # --- Инициализация логов ребалансировок ---
    calendar_rebalance_log = []
    weight_band_rebalance_log = []
    combined_rebalance_log = []
    calendar_hover_texts = []
    weight_band_hover_texts = []
    combined_hover_texts = []
    hover_dates = []
    # -------------------------------------------

    # Добавляем текст для начальной даты
    initial_hover_text = format_hover_weights(initial_holdings_template, initial_cash, price_data.iloc[0],
                                             assets_to_rebalance, all_price_assets)
    calendar_hover_texts.append(initial_hover_text)
    weight_band_hover_texts.append(initial_hover_text)
    combined_hover_texts.append(initial_hover_text)
    hover_dates.append(first_date)

    # --- 1. Логика КАЛЕНДАРНОЙ ребалансировки ---
    portfolio_cal = pd.DataFrame(index=price_data.index)
    portfolio_cal['Holdings'] = pd.Series(dtype=object)
    portfolio_cal['Cash'] = np.nan
    portfolio_cal['Total_Value'] = np.nan

    portfolio_cal.at[first_date, 'Holdings'] = initial_holdings_template.copy()
    portfolio_cal.at[first_date, 'Cash'] = initial_cash
    portfolio_cal.at[first_date, 'Total_Value'] = initial_total_value_check # Используем проверенное значение

    freq_map = {'M': 'ME', 'Q': 'QE', 'A': 'YE'}
    actual_freq = freq_map.get(rebalance_freq, rebalance_freq)
    rebalance_dates_cal_raw = pd.date_range(start=first_date, end=portfolio_cal.index[-1], freq=actual_freq)
    rebalance_dates_cal = portfolio_cal.index.intersection(rebalance_dates_cal_raw)

    for i in range(1, len(portfolio_cal.index)):
        current_date = portfolio_cal.index[i]
        prev_date = portfolio_cal.index[i-1]
        # Копируем состояние
        portfolio_cal.at[current_date, 'Holdings'] = portfolio_cal.at[prev_date, 'Holdings'].copy()
        portfolio_cal.at[current_date, 'Cash'] = portfolio_cal.at[prev_date, 'Cash']
        # Пересчитываем стоимость
        current_total_value = 0.0
        holdings_dict = portfolio_cal.at[current_date, 'Holdings']
        for ticker in all_price_assets:
            shares = holdings_dict.get(ticker, 0.0)
            price = price_data.at[current_date, ticker]
            if shares > 0 and not pd.isna(price) and price > 0:
                current_total_value += shares * price
        current_total_value += portfolio_cal.at[current_date, 'Cash']
        portfolio_cal.at[current_date, 'Total_Value'] = current_total_value
        # Ребалансировка?
        if current_date in rebalance_dates_cal:
            # (логика ребалансировки как раньше)
            new_holdings = holdings_dict.copy()
            cash_after_rebalance = 0.0
            if current_total_value > 0:
                for ticker in assets_to_rebalance:
                    target_val = current_total_value * target_weights.get(ticker, 0.0)
                    price = price_data.at[current_date, ticker]
                    shares = 0.0
                    if not pd.isna(price) and price > 0 and target_val > 0:
                        shares = target_val / price
                    new_holdings[ticker] = shares
                cash_after_rebalance = current_total_value * target_weights.get('Cash', 0.0)
            portfolio_cal.at[current_date, 'Holdings'] = new_holdings
            portfolio_cal.at[current_date, 'Cash'] = cash_after_rebalance
            # --- Логирование ---
            calendar_rebalance_log.append((current_date, 'calendar'))
            # -----------------

        # --- Логирование hover текста ПОСЛЕ всех действий за день ---
        final_holdings = portfolio_cal.at[current_date, 'Holdings']
        final_cash = portfolio_cal.at[current_date, 'Cash']
        hover_text = format_hover_weights(final_holdings, final_cash, price_data.loc[current_date],
                                          assets_to_rebalance, all_price_assets)
        calendar_hover_texts.append(hover_text)
        if len(hover_dates) <= i: hover_dates.append(current_date) # Добавляем дату, если это первый цикл
        # -----------------------------------------------------------

    calendar_rebalanced_values = portfolio_cal['Total_Value'].copy().rename('Calendar_Rebalanced_Value')

    # --- 2. Логика ребалансировки ПО ПОРОГУ ОТКЛОНЕНИЯ ДОЛИ ---
    portfolio_wb = pd.DataFrame(index=price_data.index) # wb = weight band
    portfolio_wb['Holdings'] = pd.Series(dtype=object)
    portfolio_wb['Cash'] = np.nan
    portfolio_wb['Total_Value'] = np.nan
    # Удаляем Prices_Last_Rebalance

    # Инициализация
    portfolio_wb.at[first_date, 'Holdings'] = initial_holdings_template.copy()
    portfolio_wb.at[first_date, 'Cash'] = initial_cash
    portfolio_wb.at[first_date, 'Total_Value'] = initial_total_value_check

    for i in range(1, len(portfolio_wb.index)):
        current_date = portfolio_wb.index[i]
        prev_date = portfolio_wb.index[i-1]
        # Копируем состояние
        portfolio_wb.at[current_date, 'Holdings'] = portfolio_wb.at[prev_date, 'Holdings'].copy()
        portfolio_wb.at[current_date, 'Cash'] = portfolio_wb.at[prev_date, 'Cash']
        # Пересчитываем стоимость
        current_total_value = 0.0
        holdings_dict = portfolio_wb.at[current_date, 'Holdings']
        asset_values_today = {} # Сохраняем стоимости для расчета долей
        for ticker in all_price_assets:
            shares = holdings_dict.get(ticker, 0.0)
            price = price_data.at[current_date, ticker]
            value = 0.0
            if shares > 0 and not pd.isna(price) and price > 0:
                value = shares * price
            asset_values_today[ticker] = value
            current_total_value += value
        current_total_value += portfolio_wb.at[current_date, 'Cash']
        portfolio_wb.at[current_date, 'Total_Value'] = current_total_value

        # Проверка условия ребалансировки по ДОЛЕ
        trigger_rebalance = False
        if current_total_value > 1e-9:
            for ticker in assets_to_rebalance:
                current_asset_val_ticker = asset_values_today.get(ticker, 0.0)
                current_weight = current_asset_val_ticker / current_total_value
                target_weight = target_weights.get(ticker, 0.0)

                # --- ИЗМЕНЕННАЯ ЛОГИКА ПРОВЕРКИ ОТКЛОНЕНИЯ ---
                deviation_exceeded = False
                if deviation_type == 'relative':
                    if target_weight > 1e-9:
                        relative_deviation = abs(current_weight - target_weight) / target_weight
                        if relative_deviation > threshold_weight_delta_fraction:
                            deviation_exceeded = True
                if not deviation_exceeded:
                    if abs(current_weight - target_weight) > threshold_weight_delta_fraction:
                        deviation_exceeded = True
                # -------------------------------------------------

                if deviation_exceeded:
                    trigger_rebalance = True
                    break

        # Ребалансировка, если триггер сработал
        if trigger_rebalance:
            # (стандартная логика ребалансировки, без обновления baseline цен)
            new_holdings = holdings_dict.copy()
            cash_after_rebalance = 0.0
            if current_total_value > 0:
                for ticker in assets_to_rebalance:
                    target_val = current_total_value * target_weights.get(ticker, 0.0)
                    price = price_data.at[current_date, ticker]
                    shares = 0.0
                    if not pd.isna(price) and price > 0 and target_val > 0:
                        shares = target_val / price
                    new_holdings[ticker] = shares
                cash_after_rebalance = current_total_value * target_weights.get('Cash', 0.0)
            portfolio_wb.at[current_date, 'Holdings'] = new_holdings
            portfolio_wb.at[current_date, 'Cash'] = cash_after_rebalance
            # --- Логирование ---
            weight_band_rebalance_log.append((current_date, 'weight_band'))
            # -----------------

        # --- Логирование hover текста ПОСЛЕ всех действий за день ---
        final_holdings = portfolio_wb.at[current_date, 'Holdings']
        final_cash = portfolio_wb.at[current_date, 'Cash']
        hover_text = format_hover_weights(final_holdings, final_cash, price_data.loc[current_date],
                                          assets_to_rebalance, all_price_assets)
        weight_band_hover_texts.append(hover_text)
        # -----------------------------------------------------------

    weight_band_rebalanced_values = portfolio_wb['Total_Value'].copy().rename('Weight_Band_Value') # Переименовано

    # --- 3. Логика КОМБИНИРОВАННОЙ ребалансировки (Календарь ИЛИ Доля %) ---
    portfolio_comb = pd.DataFrame(index=price_data.index)
    portfolio_comb['Holdings'] = pd.Series(dtype=object)
    portfolio_comb['Cash'] = np.nan
    portfolio_comb['Total_Value'] = np.nan
    # Удаляем Prices_Last_Price_Trigger_Rebalance

    # Инициализация
    portfolio_comb.at[first_date, 'Holdings'] = initial_holdings_template.copy()
    portfolio_comb.at[first_date, 'Cash'] = initial_cash
    portfolio_comb.at[first_date, 'Total_Value'] = initial_total_value_check

    for i in range(1, len(portfolio_comb.index)):
        current_date = portfolio_comb.index[i]
        prev_date = portfolio_comb.index[i-1]
        # Копируем состояние
        portfolio_comb.at[current_date, 'Holdings'] = portfolio_comb.at[prev_date, 'Holdings'].copy()
        portfolio_comb.at[current_date, 'Cash'] = portfolio_comb.at[prev_date, 'Cash']
        # Пересчитываем стоимость
        current_total_value = 0.0
        holdings_dict = portfolio_comb.at[current_date, 'Holdings']
        asset_values_today_comb = {} # Для расчета долей
        for ticker in all_price_assets:
            shares = holdings_dict.get(ticker, 0.0)
            price = price_data.at[current_date, ticker]
            value = 0.0
            if shares > 0 and not pd.isna(price) and price > 0:
                value = shares * price
            asset_values_today_comb[ticker] = value
            current_total_value += value
        current_total_value += portfolio_comb.at[current_date, 'Cash']
        portfolio_comb.at[current_date, 'Total_Value'] = current_total_value

        # Проверка триггеров
        calendar_trigger = current_date in rebalance_dates_cal
        weight_trigger = False
        if current_total_value > 1e-9:
            for ticker in assets_to_rebalance:
                current_asset_val_ticker = asset_values_today_comb.get(ticker, 0.0)
                current_weight = current_asset_val_ticker / current_total_value
                target_weight = target_weights.get(ticker, 0.0)

                # --- ИЗМЕНЕННАЯ ЛОГИКА ПРОВЕРКИ ОТКЛОНЕНИЯ (такая же, как в п.2) ---
                deviation_exceeded = False
                if deviation_type == 'relative':
                    if target_weight > 1e-9:
                        relative_deviation = abs(current_weight - target_weight) / target_weight
                        if relative_deviation > threshold_weight_delta_fraction:
                            deviation_exceeded = True
                if not deviation_exceeded:
                    if abs(current_weight - target_weight) > threshold_weight_delta_fraction:
                        deviation_exceeded = True
                # --------------------------------------------------------------

                if deviation_exceeded:
                    weight_trigger = True
                    break

        # Ребалансировка, если ЛЮБОЙ триггер сработал
        if (calendar_trigger or weight_trigger) and current_total_value > 1e-9:
            # (стандартная логика ребалансировки, без обновления baseline цен)
            new_holdings = holdings_dict.copy()
            cash_after_rebalance = 0.0
            if current_total_value > 0:
                for ticker in assets_to_rebalance:
                    target_val = current_total_value * target_weights.get(ticker, 0.0)
                    price = price_data.at[current_date, ticker]
                    shares = 0.0
                    if not pd.isna(price) and price > 0 and target_val > 0:
                        shares = target_val / price
                    new_holdings[ticker] = shares
                cash_after_rebalance = current_total_value * target_weights.get('Cash', 0.0)
            portfolio_comb.at[current_date, 'Holdings'] = new_holdings
            portfolio_comb.at[current_date, 'Cash'] = cash_after_rebalance
            # Удалено обновление цен триггера
            # --- Логирование ---
            event_type = 'calendar' if calendar_trigger else 'weight_band'
            combined_rebalance_log.append((current_date, event_type))
            # -------------------
            # (логика обновления baseline цен для weight_trigger)
            if weight_trigger:
                # ... обновление Prices_Last_Price_Trigger_Rebalance
                pass # Оставил pass, так как сам код не меняется

        # --- Логирование hover текста ПОСЛЕ всех действий за день ---
        final_holdings = portfolio_comb.at[current_date, 'Holdings']
        final_cash = portfolio_comb.at[current_date, 'Cash']
        hover_text = format_hover_weights(final_holdings, final_cash, price_data.loc[current_date],
                                          assets_to_rebalance, all_price_assets)
        combined_hover_texts.append(hover_text)
        # -----------------------------------------------------------

    combined_rebalanced_values = portfolio_comb['Total_Value'].copy().rename('Combined_Value')

    # --- 4. Логика Buy & Hold (на основе целевых весов) --- (без изменений)
    initial_prices = price_data.iloc[0]
    bh_target_assets = [col for col in target_weights.keys() if col != 'Cash']
    initial_investment_per_target_asset = {ticker: initial_capital * target_weights.get(ticker, 0.0) for ticker in bh_target_assets}
    initial_target_shares = {}
    for ticker in bh_target_assets:
         price = initial_prices.get(ticker, np.nan)
         if not pd.isna(price) and price > 0:
             initial_target_shares[ticker] = initial_investment_per_target_asset[ticker] / price
         else:
             initial_target_shares[ticker] = 0.0
    cash_bh_target = initial_capital * target_weights.get('Cash', 0.0)
    bh_target_values = pd.Series(index=price_data.index, dtype=float)
    asset_prices_bh_target = price_data[bh_target_assets]
    portfolio_asset_values_bh_target = asset_prices_bh_target.mul(pd.Series(initial_target_shares), axis=1)
    bh_target_values = portfolio_asset_values_bh_target.sum(axis=1) + cash_bh_target
    bh_target_values.rename('BH_Target_Value', inplace=True)

    # --- 5. Логика Buy & Hold (для каждого актива отдельно) --- (без изменений)
    individual_bh_results = {}
    all_tickers = [col for col in price_data.columns if col != 'Cash']
    for ticker in all_tickers:
        initial_price = initial_prices.get(ticker, np.nan)
        if pd.isna(initial_price) or initial_price <= 0:
            bh_individual_values = pd.Series(np.nan, index=price_data.index, name=f"BH_{ticker}")
        else:
            initial_shares_individual = initial_capital / initial_price
            bh_individual_values = price_data[ticker] * initial_shares_individual
            bh_individual_values.rename(f"BH_{ticker}", inplace=True)
        individual_bh_results[f"BH_{ticker}"] = bh_individual_values

    # Собираем все результаты СТОИМОСТЕЙ
    all_results_list = [calendar_rebalanced_values, weight_band_rebalanced_values, combined_rebalanced_values, bh_target_values] + list(individual_bh_results.values())
    results = pd.concat(all_results_list, axis=1)
    results.ffill(inplace=True)

    # --- Расчет рядов ПРОСАДОК для основных стратегий ---
    drawdown_results_dict = {}
    # Обновляем список, используя Weight_Band_Value
    main_strategy_cols = ['Calendar_Rebalanced_Value', 'Weight_Band_Value', 'Combined_Value', 'BH_Target_Value']
    for col in main_strategy_cols:
        if col in results.columns:
            series = pd.to_numeric(results[col], errors='coerce').dropna()
            if not series.empty:
                 drawdown_results_dict[col] = calculate_drawdown_series(series)
            else:
                 drawdown_results_dict[col] = pd.Series(np.nan, index=results.index, name=col)
        else:
            drawdown_results_dict[col] = pd.Series(np.nan, index=results.index, name=col)

    drawdown_results = pd.concat(drawdown_results_dict, axis=1)

    # --- Формирование словаря логов --- 
    rebalance_log = {
        'Calendar_Rebalanced_Value': calendar_rebalance_log,
        'Weight_Band_Value': weight_band_rebalance_log,
        'Combined_Value': combined_rebalance_log
    }
    # ------------------------------------

    # --- Формирование DataFrame для hover текста --- 
    hover_weights_text = pd.DataFrame({
        'Calendar_Rebalanced_Value': calendar_hover_texts,
        'Weight_Band_Value': weight_band_hover_texts,
        'Combined_Value': combined_hover_texts
    }, index=hover_dates) # Используем собранные даты как индекс
    # ---------------------------------------------

    # Возвращаем ЧЕТЫРЕ элемента
    return results, drawdown_results, rebalance_log, hover_weights_text

# --- Функции расчета метрик ---

def _calculate_cagr(series: pd.Series) -> float:
    """Рассчитывает CAGR для временного ряда стоимости."""
    if series.empty or pd.isna(series.iloc[0]) or series.iloc[0] == 0:
        return np.nan
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if pd.isna(end_value):
        # Если конечного значения нет, попробуем взять предпоследнее non-NA
        valid_series = series.dropna()
        if valid_series.empty:
             return np.nan
        end_value = valid_series.iloc[-1]
        end_date = valid_series.index[-1]
    else:
        end_date = series.index[-1]

    start_date = series.index[0]
    num_years = (end_date - start_date).days / 365.25
    if num_years <= 0:
        return np.nan
    if end_value <= 0 and start_value > 0:
        return -1.0
    if end_value < 0 and start_value < 0:
         return np.nan # Неопределенность
    # Избегаем деления на ноль или отрицательный старт
    if start_value <= 0:
         return np.nan
    ratio = end_value / start_value
    if ratio < 0: # Не может быть отрицательного отношения для CAGR
         return np.nan
    cagr = ratio ** (1 / num_years) - 1
    return cagr

def _calculate_max_drawdown(series: pd.Series) -> (float, float, float):
    """Рассчитывает максимальную просадку (в %), абсолютную просадку и пик перед ней."""
    if series.empty or series.isna().all():
        return np.nan, np.nan, np.nan
    series = series.dropna()
    if series.empty:
         return np.nan, np.nan, np.nan
    cumulative_max = series.cummax()
    # Заменяем 0 на NaN, чтобы избежать деления на ноль, если серия начинается с 0 или имеет 0
    safe_cumulative_max = cumulative_max.replace(0, np.nan)
    drawdown = (series - safe_cumulative_max) / safe_cumulative_max
    drawdown = drawdown.fillna(0) # Заполняем NaN (возникающие из-за safe_cumulative_max) нулями

    if drawdown.empty:
         return 0.0, 0.0, series.iloc[0] if not series.empty else np.nan

    max_drawdown_pct = drawdown.min()
    # Если все просадки 0, то ищем первый индекс
    try:
         idx_min = drawdown.idxmin()
    except ValueError: # Может возникнуть, если все значения одинаковы
         idx_min = series.index[0]

    peak_before_max_drawdown = cumulative_max.loc[idx_min] if idx_min in cumulative_max.index else series.iloc[0]

    if pd.isna(max_drawdown_pct) or pd.isna(peak_before_max_drawdown):
         max_drawdown_abs = 0.0 # Или np.nan? Если pct NaN, то и abs NaN
    else:
         max_drawdown_abs = max_drawdown_pct * peak_before_max_drawdown

    return max_drawdown_pct, max_drawdown_abs, peak_before_max_drawdown


def _calculate_volatility(series: pd.Series) -> float:
    """Рассчитывает аннуализированную волатильность."""
    if series.empty or series.isna().all() or len(series) < 2:
        return np.nan
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return np.nan
    # Обработка inf/-inf, если они вдруг появятся
    if np.isinf(daily_returns).any():
         print("Warning: Inf values detected in daily returns for Volatility calculation.")
         daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
         if daily_returns.empty:
              return np.nan
    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return volatility

def _calculate_sharpe(series: pd.Series, risk_free_rate_annual: float) -> float:
    """Рассчитывает коэффициент Шарпа (аннуализированный)."""
    if series.empty or series.isna().all() or len(series) < 2:
        return np.nan
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return np.nan
    # Обработка inf/-inf
    if np.isinf(daily_returns).any():
         print("Warning: Inf values detected in daily returns for Sharpe calculation.")
         daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
         if daily_returns.empty:
              return np.nan
    # Проверка на нулевое стандартное отклонение
    std_dev = daily_returns.std()
    if std_dev < 1e-10 or pd.isna(std_dev):
         # Если стандартное отклонение 0, Шарп не определен, если только средняя доходность не равна безрисковой
         mean_return = daily_returns.mean()
         risk_free_daily = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
         if abs(mean_return - risk_free_daily) < 1e-10:
              return 0.0 # Или np.nan? Часто считают 0, если нет избыточной доходности и риска
         else:
              # Бесконечный Шарп? Возвращаем NaN как неопределенность
              return np.nan
    excess_returns = daily_returns - (risk_free_rate_annual / TRADING_DAYS_PER_YEAR)
    sharpe_ratio = np.mean(excess_returns) / std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe_ratio

def _calculate_sortino(series: pd.Series, risk_free_rate_annual: float) -> float:
    """Рассчитывает коэффициент Сортино (аннуализированный)."""
    if series.empty or series.isna().all() or len(series) < 2:
        return np.nan
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return np.nan
    # Обработка inf/-inf
    if np.isinf(daily_returns).any():
         print("Warning: Inf values detected in daily returns for Sortino calculation.")
         daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
         if daily_returns.empty:
              return np.nan

    target_return_daily = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    downside_returns = daily_returns[daily_returns < target_return_daily]

    # Проверяем, есть ли отрицательные отклонения
    if downside_returns.empty:
        # Если нет доходностей ниже целевой, Сортино не определен или бесконечен?
        # Если средняя доходность > целевой, то риск = 0, можно вернуть inf или nan
        # Если средняя доходность <= целевой, но все >= target, можно вернуть 0?
        # Часто возвращают NaN в этом случае.
        return np.nan

    downside_deviation = np.sqrt(np.mean((downside_returns - target_return_daily)**2))

    # Проверка на нулевое отклонение вниз
    if downside_deviation < 1e-10 or pd.isna(downside_deviation):
        # Если риск = 0, Сортино не определен, если только избыточная доходность не 0
        mean_excess_return = np.mean(daily_returns - target_return_daily)
        if abs(mean_excess_return) < 1e-10:
             return 0.0 # Нет изб. доходности, нет риска -> 0
        else:
             # Положительная изб. доходность при нулевом риске -> Бесконечность? Вернем NaN.
             return np.nan

    mean_excess_return = np.mean(daily_returns - target_return_daily)
    sortino_ratio = mean_excess_return / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino_ratio


def _calculate_recovery_factor(series: pd.Series, max_drawdown_abs: float) -> float:
     """Рассчитывает фактор восстановления."""
     if series.empty or series.isna().all():
         return np.nan
     series = series.dropna()
     if series.empty or len(series) < 2:
         return np.nan
     # max_drawdown_abs должен быть отрицательным или 0
     if pd.isna(max_drawdown_abs) or max_drawdown_abs > 1e-10 : # Проверка, что не положительный
         # Если просадки не было (max_drawdown_abs ~ 0) или она не рассчитана, фактор не определен
         return np.nan
     if abs(max_drawdown_abs) < 1e-10: # Если просадка 0
        # Если есть прибыль, фактор восстановления бесконечен? Вернем NaN
        # Если прибыли нет, фактор 0/0? Вернем NaN
        return np.nan

     total_profit = series.iloc[-1] - series.iloc[0]
     if pd.isna(total_profit):
          return np.nan
     # Делим на абсолютную величину просадки
     recovery_factor = total_profit / abs(max_drawdown_abs)
     return recovery_factor


@st.cache_data
def calculate_metrics(backtest_results: pd.DataFrame, risk_free_rate_annual: float,
                        rebalance_log: Dict[str, List[Tuple[Timestamp, str]]]) -> Dict[str, Dict[str, float]]:
    """Рассчитывает основные метрики производительности для каждой стратегии.

    Args:
        backtest_results (pd.DataFrame): DataFrame со стоимостью портфелей.
        risk_free_rate_annual (float): Годовая безрисковая ставка (десятичная дробь).
        rebalance_log (Dict[str, List[Tuple[Timestamp, str]]]): Лог ребалансировок.

    Returns:
        Dict[str, Dict[str, float]]: Словарь метрик для каждой стратегии.
    """
    metrics = {}
    internal_names_map = {
        'Ребаланс (Календарь)': 'Calendar_Rebalanced_Value',
        'Ребаланс (Доля %)': 'Weight_Band_Value', # Обновлено для соответствия новому имени
        'Ребаланс (Комби)': 'Combined_Value',
        'B&H (Целевые веса)': 'BH_Target_Value'
    }

    # Добавляем B&H для отдельных тикеров в карту (динамически)
    for col in backtest_results.columns:
        if col.startswith('BH_') and col != 'BH_Target_Value':
            ticker_name = col.split('_', 1)[1]
            display_name = f"B&H ({ticker_name})"
            internal_names_map[display_name] = col

    for name, internal_col_name in internal_names_map.items():
        if internal_col_name in backtest_results.columns:
            series = pd.to_numeric(backtest_results[internal_col_name], errors='coerce').dropna()
            if series.empty:
                metrics[name] = {k: np.nan for k in [
                    'Start Value', 'End Value', 'CAGR', 'Max Drawdown %', 'Max Drawdown Abs $', 'Peak Before MDD',
                    'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Recovery Factor', 'Rebalance Count' # Вернули Rebalance Count
                ]}
                continue

            start_value = series.iloc[0] if not series.empty else np.nan
            end_value = series.iloc[-1] if not series.empty else np.nan
            cagr = _calculate_cagr(series)
            mdd_pct, mdd_abs, peak_before_mdd = _calculate_max_drawdown(series)
            volatility = _calculate_volatility(series)
            sharpe = _calculate_sharpe(series, risk_free_rate_annual)
            sortino = _calculate_sortino(series, risk_free_rate_annual)
            recovery_factor = _calculate_recovery_factor(series, mdd_abs)

            rebalance_count = np.nan # Инициализируем как NaN
            # --- Добавляем расчет количества ребалансировок ---
            if internal_col_name and internal_col_name in rebalance_log: # Проверяем, есть ли лог для этой стратегии
                rebalance_count = len(rebalance_log[internal_col_name])
            elif name.startswith('B&H'): # Для стратегий B&H ставим 0
                 rebalance_count = 0
            # ---------------------------------------------------

            metrics[name] = {
                'Start Value': start_value,
                'End Value': end_value,
                'CAGR': cagr,
                'Max Drawdown %': mdd_pct * 100 if not pd.isna(mdd_pct) else np.nan,
                'Max Drawdown Abs $': mdd_abs,
                'Peak Before MDD': peak_before_mdd,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe,
                'Sortino Ratio': sortino,
                'Recovery Factor': recovery_factor,
                'Rebalance Count': rebalance_count # Добавляем метрику
            }
        else:
            metrics[name] = {k: np.nan for k in [
                'Start Value', 'End Value', 'CAGR', 'Max Drawdown %', 'Max Drawdown Abs $', 'Peak Before MDD',
                'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Recovery Factor', 'Rebalance Count' # Вернули Rebalance Count
            ]}

    return metrics 
