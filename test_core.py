import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from core import run_backtest # Импортируем функцию для тестирования

# --- Фикстура для тестовых данных ---
@pytest.fixture
def mock_price_data():
    """Создает DataFrame с предсказуемыми ценами для тестов."""
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='B') # Рабочие дни
    data = {
        'SPY': np.linspace(100, 110, len(dates)), # Плавный рост SPY
        'GLD': np.concatenate([np.linspace(100, 105, len(dates)//2), # Медленный рост GLD
                               np.linspace(105, 130, len(dates) - len(dates)//2)]), # Резкий рост GLD во 2й половине
        'Cash': 1.0
    }
    prices = pd.DataFrame(data, index=dates)
    return prices

# --- Новая фикстура для зеркальных данных ---
@pytest.fixture
def mirrored_price_data():
    """Создает DataFrame с зеркально изменяющимися ценами."""
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='B') # 3 месяца
    num_days = len(dates)
    data = {
        'UP': np.linspace(100, 120, num_days), # Рост с 100 до 120
        'DOWN': np.linspace(100, 80, num_days),  # Падение с 100 до 80
        'Cash': 1.0
    }
    prices = pd.DataFrame(data, index=dates)
    return prices

# --- Тесты ---

def test_basic_rebalance_vs_bh(mock_price_data):
    """
    Тест базового сценария: 60% SPY, 40% GLD, ежемесячная ребалансировка.
    Проверяем конечные стоимости и факт, что они отличаются.
    """
    price_data = mock_price_data
    target_weights = {'SPY': 0.6, 'GLD': 0.4, 'Cash': 0.0}
    rebalance_freq = 'M' # Ежемесячно
    initial_capital = 10000.0

    results = run_backtest(price_data, target_weights, rebalance_freq, initial_capital)

    assert results is not None, "Результат бэктеста не должен быть None"
    assert not results.empty, "Результаты бэктеста не должны быть пустыми"
    assert 'Rebalanced_Value' in results.columns
    assert 'BH_Value' in results.columns
    assert not results['Rebalanced_Value'].isna().all(), "Столбец Rebalanced_Value не должен содержать только NaN"
    assert not results['BH_Value'].isna().all(), "Столбец BH_Value не должен содержать только NaN"

    # Проверяем начальные значения
    assert results['Rebalanced_Value'].iloc[0] == pytest.approx(initial_capital)
    assert results['BH_Value'].iloc[0] == pytest.approx(initial_capital)

    # Проверяем конечные значения (ожидаем, что они будут разными из-за резкого роста GLD)
    final_rebalanced = results['Rebalanced_Value'].iloc[-1]
    final_bh = results['BH_Value'].iloc[-1]

    # Примерные ручные расчеты для B&H:
    # Начальные доли: SPY = 6000/100 = 60, GLD = 4000/100 = 40
    # Конечные цены: SPY ~ 110, GLD ~ 130
    # Конечная стоимость B&H: 60 * 110 + 40 * 130 = 6600 + 5200 = 11800
    spy_shares_bh = target_weights['SPY'] * initial_capital / price_data['SPY'].iloc[0]
    gld_shares_bh = target_weights['GLD'] * initial_capital / price_data['GLD'].iloc[0]
    expected_final_bh = spy_shares_bh * price_data['SPY'].iloc[-1] + gld_shares_bh * price_data['GLD'].iloc[-1]
    assert final_bh == pytest.approx(expected_final_bh)

    # Для ребалансировки сложнее посчитать вручную, но она должна отличаться от B&H
    # В данном случае, B&H выиграет, т.к. доля сильно выросшего GLD не уменьшалась
    assert final_rebalanced != final_bh
    # Ожидаем, что B&H будет больше, так как GLD сильно вырос, а ребалансировка продавала бы его
    assert final_bh > final_rebalanced

    print(f"\nTest Basic Rebalance vs B&H:")
    print(f"Final Rebalanced Value: {final_rebalanced:.2f}")
    print(f"Final Buy & Hold Value: {final_bh:.2f}")

def test_zero_weight_asset(mock_price_data):
    """
    Тест сценария с нулевым весом для GLD.
    Ожидаем, что и Rebalanced, и BH будут отслеживать только SPY.
    """
    price_data = mock_price_data
    # Вес GLD = 0!
    target_weights = {'SPY': 1.0, 'GLD': 0.0, 'Cash': 0.0}
    rebalance_freq = 'M'
    initial_capital = 10000.0

    results = run_backtest(price_data, target_weights, rebalance_freq, initial_capital)

    assert results is not None
    assert not results.empty
    assert 'Rebalanced_Value' in results.columns
    assert 'BH_Value' in results.columns

    # Ожидаемый результат для B&H: только SPY
    initial_spy_shares = initial_capital / price_data['SPY'].iloc[0]
    expected_bh_values = initial_spy_shares * price_data['SPY']
    # Сравниваем весь столбец с ожидаемым поведением SPY
    # Используем assert_series_equal для более точного сравнения временных рядов
    pd.testing.assert_series_equal(results['BH_Value'], expected_bh_values, check_names=False, rtol=1e-5)

    # Ожидаемый результат для Rebalanced: тоже только SPY, т.к. вес GLD всегда 0
    # В этом случае ребалансировка не должна ничего менять, т.к. 100% в SPY
    pd.testing.assert_series_equal(results['Rebalanced_Value'], expected_bh_values, check_names=False, rtol=1e-5)

    print(f"\nTest Zero Weight Asset (GLD=0):")
    print(f"Final Rebalanced Value: {results['Rebalanced_Value'].iloc[-1]:.2f}")
    print(f"Final Buy & Hold Value: {results['BH_Value'].iloc[-1]:.2f}")
    print(f"Expected Final Value (SPY only): {expected_bh_values.iloc[-1]:.2f}")

def test_rebalance_with_cash(mock_price_data):
    """
    Тест сценария с долей кэша (20%).
    Проверяем, что кэш учитывается и конечные стоимости отличаются от B&H.
    """
    price_data = mock_price_data
    # Вес кэша = 20%
    target_weights = {'SPY': 0.5, 'GLD': 0.3, 'Cash': 0.2}
    rebalance_freq = 'M'
    initial_capital = 10000.0

    results = run_backtest(price_data, target_weights, rebalance_freq, initial_capital)

    assert results is not None
    assert not results.empty
    assert 'Rebalanced_Value' in results.columns
    assert 'BH_Value' in results.columns

    # Проверяем начальные значения
    assert results['Rebalanced_Value'].iloc[0] == pytest.approx(initial_capital)
    assert results['BH_Value'].iloc[0] == pytest.approx(initial_capital)

    # Проверяем конечные значения
    final_rebalanced = results['Rebalanced_Value'].iloc[-1]
    final_bh = results['BH_Value'].iloc[-1]

    # Примерный расчет B&H с кэшем:
    initial_cash_bh = initial_capital * target_weights['Cash']
    spy_shares_bh = target_weights['SPY'] * initial_capital / price_data['SPY'].iloc[0]
    gld_shares_bh = target_weights['GLD'] * initial_capital / price_data['GLD'].iloc[0]
    expected_final_bh = spy_shares_bh * price_data['SPY'].iloc[-1] + gld_shares_bh * price_data['GLD'].iloc[-1] + initial_cash_bh
    assert final_bh == pytest.approx(expected_final_bh)

    # Ожидаем, что результаты будут отличаться
    assert final_rebalanced != final_bh
    # B&H снова должен быть немного лучше из-за сильного роста GLD и фиксированного кэша
    assert final_bh > final_rebalanced

    # Дополнительно можно проверить, что кэш в ребалансировке не равен нулю в конце
    # (Это сложнее сделать точно без доступа к внутреннему состоянию portfolio,
    # но сам факт отличия final_rebalanced от final_bh и от сценария без кэша показателен)

    print(f"\nTest Rebalance with Cash (20%):")
    print(f"Final Rebalanced Value: {final_rebalanced:.2f}")
    print(f"Final Buy & Hold Value: {final_bh:.2f}")

# Добавим остальные тесты ниже...

def test_rebalance_mirrored_data(mirrored_price_data):
    """
    Тест на зеркальных данных (UP 50%, DOWN 50%).
    Ожидаем, что стоимость ребалансированного портфеля останется ~константной.
    """
    price_data = mirrored_price_data
    target_weights = {'UP': 0.5, 'DOWN': 0.5, 'Cash': 0.0}
    rebalance_freq = 'M' # Ежемесячно
    initial_capital = 10000.0

    results = run_backtest(price_data, target_weights, rebalance_freq, initial_capital)

    assert results is not None
    assert not results.empty
    assert 'Rebalanced_Value' in results.columns
    assert 'BH_Value' in results.columns

    # Проверяем начальные значения
    assert results['Rebalanced_Value'].iloc[0] == pytest.approx(initial_capital)
    assert results['BH_Value'].iloc[0] == pytest.approx(initial_capital)

    # Проверяем конечные значения
    final_rebalanced = results['Rebalanced_Value'].iloc[-1]
    final_bh = results['BH_Value'].iloc[-1]

    # Ожидаемая конечная стоимость B&H = 10000
    expected_final_bh = 10000.0
    assert final_bh == pytest.approx(expected_final_bh)

    # Ожидаемая конечная стоимость ребалансировки должна быть ОЧЕНЬ близка к 10000
    # Допускаем небольшое отклонение из-за дискретности ребалансировки
    assert final_rebalanced == pytest.approx(initial_capital, abs=50) # Допуск +/- 50 у.е.

    # Также проверим, что максимальное отклонение в течение периода невелико
    max_deviation = (results['Rebalanced_Value'] - initial_capital).abs().max()
    assert max_deviation < 100 # Допуск +/- 100 у.е. в течение периода

    print(f"\nTest Mirrored Data (50/50):")
    print(f"Final Rebalanced Value: {final_rebalanced:.2f}")
    print(f"Final Buy & Hold Value: {final_bh:.2f}")
    print(f"Max deviation from initial capital during period: {max_deviation:.2f}") 