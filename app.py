import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import core # Импортируем наш модуль с логикой
import plots # Импортируем наш модуль с графиками
from typing import Dict, List

# --- Настройка страницы ---
st.set_page_config(layout="wide", page_title="Бэктестер Ребалансировки")

# --- CSS для уменьшения шрифтов И ДОБАВЛЕНИЯ РАМОК КОНТЕЙНЕРАМ И ВЫРАВНИВАНИЯ В КОЛОНКАХ ---
st.markdown("""
<style>
    /* Уменьшаем шрифт для опций и выбранных элементов в multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        font-size: 0.7em !important; /* ЕЩЕ МЕНЬШЕ */
        /* background-color: #eee !important; /* Стандартный цвет */
        /* color: #333 !important; */
        /* border-radius: 0.25rem !important; */
        /* padding: 0.15rem 0.4rem !important; */
    }
    div[data-baseweb="select"] > div {
        font-size: 0.8em !important; /* ЕЩЕ МЕНЬШЕ */
    }
    /* Уменьшаем шрифт для опций в выпадающем списке */
     div[data-baseweb="popover"] li {
        font-size: 0.8em !important; /* ЕЩЕ МЕНЬШЕ */
     }
     /* Уменьшаем шрифт для подписей к весам (уже есть small, сделаем чуть меньше) */
    .stNumberInput small {
         font-size: 0.75em !important;
     }

    /* --- Стили для рамок контейнеров --- */
    /* Выбираем контейнеры верхнего уровня в основной области */
    .main .stBlock > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid #e0e0e0; /* Светло-серая рамка */
        border-radius: 5px;       /* Скругленные углы */
        padding: 1.5rem;            /* Внутренний отступ */
        margin-bottom: 1.5rem;      /* Внешний отступ снизу */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Легкая тень */
    }

    /* --- CSS для вертикального выравнивания в колонках --- */
    div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        justify-content: center; /* Оставляем попытку центрирования */
    }

    /* --- УБИРАЕМ CSS для метки st.toggle, так как метка будет скрыта --- */
    /* div[data-testid="stToggle"] label p {
        font-size: 0.9em !important;
        margin-bottom: 0px !important;
    } */
</style>
""", unsafe_allow_html=True)

# Уменьшенный заголовок
st.subheader("Интерактивный Бэктестер Ребалансировки Портфеля")

# --- Определения активов, групп и цветов ---

# Расширенный словарь тикеров
COMMON_TICKERS = {
    # Акции США
    'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'IWM': 'Russell 2000 ETF', 'DIA': 'Dow Jones ETF',
    '^GSPC': 'S&P 500 Index',
    'XLK': 'Technology Sector SPDR Fund', 'XLF': 'Financial Sector SPDR Fund',
    'XLP': 'Consumer Staples Sel Sec SPDR',
    'XLV': 'Healthcare Sector SPDR Fund',
    # Акции США (Крупные компании)
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc. (Class A)', 'AMZN': 'Amazon.com, Inc.',
    'BRK-B': 'Berkshire Hathaway Inc. (Cl B)',
    'NVDA': 'NVIDIA Corporation', 'MCD': "McDonald's Corporation", 'KO': 'The Coca-Cola Company',
    'JNJ': 'Johnson & Johnson', 'JPM': 'JPMorgan Chase & Co.',
    # Акции Развитые страны
    'EFA': 'MSCI EAFE ETF', 'VEA': 'FTSE Developed Markets ETF',
    # Акции Развивающиеся рынки
    'EEM': 'MSCI Emerging Markets ETF', 'VWO': 'FTSE Emerging Markets ETF',
    # Облигации США (Казначейские)
    'IEF': '7-10 Year Treasury Bond ETF', 'TLT': '20+ Year Treasury Bond ETF', 'SHY': '1-3 Year Treasury Bond ETF',
    'TIP': 'TIPS Bond ETF',
    # Облигации США (Корпоративные/Совокупные)
    'AGG': 'US Aggregate Bond ETF', 'LQD': 'Investment Grade Corporate Bond ETF', 'HYG': 'High Yield Corporate Bond ETF',
    # Облигации Международные / Развивающиеся рынки
    'BNDX': 'Total Intl Bond ETF (hedged)',
    'EMB': 'USD Emerging Markets Bond ETF',
    # Золото
    'GLD': 'Gold ETF',
    '^XAU': 'PHLX Gold/Silver Index',
    # Недвижимость
    'VNQ': 'Real Estate ETF',
    # Сырье
    'DBC': 'Commodity Index Tracking Fund',
    # Криптовалюты
    'BTC-USD': 'Bitcoin USD',
    'ETH-USD': 'Ethereum USD',
    'LTC-USD': 'Litecoin USD',
    # Валюты
    'FXE': 'Euro ETF (vs USD)',
    'FXY': 'Japanese Yen ETF (vs USD)',
    'FXB': 'British Pound ETF (vs USD)'
}

# Группы активов (сопоставление тикера с названием группы)
ASSET_GROUPS = {
    'SPY': 'Акции США', 'QQQ': 'Акции США', 'IWM': 'Акции США', 'DIA': 'Акции США',
    '^GSPC': 'Акции США (Индекс)',
    'XLK': 'Акции США (Сектор)', 'XLF': 'Акции США (Сектор)',
    'XLP': 'Акции США (Сектор)',
    'XLV': 'Акции США (Сектор)',
    'AAPL': 'Акции США (Крупн. комп.)', 'MSFT': 'Акции США (Крупн. комп.)', 'GOOGL': 'Акции США (Крупн. комп.)', 'AMZN': 'Акции США (Крупн. комп.)',
    'BRK-B': 'Акции США (Крупн. комп.)',
    'NVDA': 'Акции США (Крупн. комп.)', 'MCD': 'Акции США (Крупн. комп.)', 'KO': 'Акции США (Крупн. комп.)',
    'JNJ': 'Акции США (Крупн. комп.)', 'JPM': 'Акции США (Крупн. комп.)',
    'EFA': 'Акции Разв.', 'VEA': 'Акции Разв.',
    'EEM': 'Акции Развивш.', 'VWO': 'Акции Развивш.',
    'IEF': 'Обл. США (Казн.)', 'TLT': 'Обл. США (Казн.)', 'SHY': 'Обл. США (Казн.)',
    'TIP': 'Обл. США (TIPS)',
    'AGG': 'Обл. США (Сумм.)', 'LQD': 'Обл. США (Корп.)', 'HYG': 'Обл. США (ВДО)',
    'BNDX': 'Обл. Межд.',
    'EMB': 'Обл. Развивш.',
    'GLD': 'Золото',
    '^XAU': 'Золото (Индекс комп.)',
    'VNQ': 'Недвиж.',
    'DBC': 'Сырье',
    'BTC-USD': 'Крипто', 'ETH-USD': 'Крипто', 'LTC-USD': 'Крипто',
    'FXE': 'Валюты', 'FXY': 'Валюты', 'FXB': 'Валюты',
    'Cash': 'Кэш'
}

# Цвета для групп (можно настроить) - используем палитру Plotly по умолчанию
import plotly.express as px
# Генерируем достаточное количество цветов из стандартной палитры
unique_groups = sorted(list(set(ASSET_GROUPS.values())))
colors = px.colors.qualitative.Plotly # или Set3, Pastell, etc.
GROUP_COLORS = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
GROUP_COLORS['Кэш'] = '#AAAAAA' # Серый для кэша

# --- Генерация опций для multiselect в формате "ТИКЕР - Название" ---
options_formatted = [f"{ticker} - {name}" for ticker, name in COMMON_TICKERS.items()]

# --- Callback для обновления весов ---
def adjust_weights():
    # Получаем текущие выбранные ФОРМАТИРОВАННЫЕ строки из состояния мултиселекта
    current_selected_formatted = st.session_state.selected_assets_multiselect
    # Парсим тикеры из форматированных строк
    current_tickers = [s.split(' - ')[0] for s in current_selected_formatted]
    # Список для установки весов
    assets_for_weighting = current_tickers + ['Cash']
    num_assets_plus_cash = len(assets_for_weighting)
    new_default_weight = 100.0 / num_assets_plus_cash if num_assets_plus_cash > 0 else 0

    # Обновляем веса в session_state для текущего выбора (используем ТИКЕРЫ как ключ)
    for ticker in assets_for_weighting: # Используем тикеры
        st.session_state[f"weight_{ticker}"] = new_default_weight # Ключ по тикеру

# --- Боковая панель для ввода параметров ---
# Убираем общий заголовок, он будет в expander
# st.sidebar.header("Параметры Бэктеста")

with st.sidebar.expander("Основные параметры", expanded=True): # Раскрыт по умолчанию

    # Ввод тикеров через multiselect
    default_tickers = ['SPY', 'GLD', 'IEF', 'BTC-USD'] # Оставляем тикеры как есть
    # Генерируем форматированные названия по умолчанию
    default_names_formatted = [f"{t} - {COMMON_TICKERS.get(t, t)}" for t in default_tickers]

    # Инициализируем веса по умолчанию в session_state при первом запуске
    if 'selected_assets_multiselect' not in st.session_state:
        st.session_state.selected_assets_multiselect = default_names_formatted # Сохраняем ФОРМАТИРОВАННЫЕ имена
        adjust_weights() # Вызываем для установки начальных весов по ТИКЕРАМ

    selected_names_formatted = st.multiselect( # Убрал st.sidebar, т.к. уже внутри with
        "Выберите активы",
        options=options_formatted, # Используем форматированные опции
        key="selected_assets_multiselect", # Ключ для доступа к состоянию и для callback
        on_change=adjust_weights # Назначаем callback
    )

    # Получаем ТИКЕРЫ из выбранных форматированных строк
    tickers = [s.split(' - ')[0] for s in selected_names_formatted]
    # Если ничего не выбрано, используем по умолчанию
    if not tickers:
        tickers = ['SPY', 'GLD', 'IEF', 'BTC-USD']

    # Создаем карту тикер -> название для выбранных (из COMMON_TICKERS по тикерам)
    selected_ticker_map = {ticker: COMMON_TICKERS.get(ticker, ticker) for ticker in tickers}

    # Создаем словарь цветов для выбранных тикеров
    selected_ticker_colors = {
        ticker: GROUP_COLORS.get(ASSET_GROUPS.get(ticker, 'Другое'), '#333333')
        for ticker in tickers
    }

    # Ввод дат
    min_date = dt.date(1990, 1, 1)  # Set the earliest possible date
    today = dt.date.today()
    start_date = st.date_input(
        "Начальная дата",
        value=dt.date(2010, 1, 1),  # Default start date
        min_value=min_date,        # Set minimum allowed date
        max_value=today - dt.timedelta(days=1), # Max is yesterday
        help="Select the start date for the historical data.",
    )
    end_date = st.date_input(
        "Конечная дата",
        value=today,
        min_value=start_date + dt.timedelta(days=1), # Min is day after start
        max_value=today,           # Max is today
        help="Select the end date for the historical data.",
    )

    # Выбор частоты ребалансировки
    rebalance_freq_options = {'ME': 'Ежемесячно', 'QE': 'Ежеквартально', 'YE': 'Ежегодно'} # Обновлено на новые коды
    rebalance_freq_display = st.selectbox( # Убрал st.sidebar
        "Частота ребалансировки (Календарная)", # Добавил уточнение
        options=list(rebalance_freq_options.keys()),
        format_func=lambda x: rebalance_freq_options[x]
    )

    # --- ИЗМЕНЕННЫЙ БЛОК: Порог и Тип отклонения в колонках ---
    # Добавляем общий заголовок
    st.markdown("<small>Порог %, абс/отн</small>", unsafe_allow_html=True) # Используем <small> для заголовка
    # Меняем пропорции и скрываем индивидуальные метки
    col1, col2 = st.columns([3, 1]) # Пропорции: 3 части для ввода, 1 для переключателя

    with col1:
        price_change_threshold = st.number_input(
            "Порог откл. (%)", # Метка нужна для streamlit, но будет скрыта
            label_visibility="collapsed", # Скрываем метку
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="Порог отклонения доли актива от целевой (абсолютный или относительный).", # Обновляем help
            key='price_change_threshold_input'
        )

    with col2:
        use_relative_threshold = st.toggle(
            "Относительный порог?", # Метка нужна для streamlit, но будет скрыта
            label_visibility="collapsed", # Скрываем метку
            value=False,
            help="Выкл = абсолютный порог (%), Вкл = относительный порог (%).", # Обновляем help
            key='deviation_type_toggle'
        )
        deviation_type = 'relative' if use_relative_threshold else 'absolute'
    # -------------------------------------------------------

    # --- ДОБАВЛЯЕМ ОТОБРАЖЕНИЕ РЕЖИМА --- 
    # st.caption(f"Режим загрузки данных: {core.DEFAULT_LOADING_MODE}")

    # Безрисковая ставка
    rf_rate_percent = st.number_input("Безрисковая ставка (% годовых)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    rf_rate_decimal = rf_rate_percent / 100.0

    # Начальный капитал (вне expander, т.к. он не меняется)
    initial_capital = 100000.0
    st.sidebar.markdown(f"**Начальный капитал:** `${initial_capital:,.0f}`")

    # Ввод весов в отдельном expander
    with st.sidebar.expander("Целевые веса (%)", expanded=True): # Раскрыт по умолчанию
        # Убираем st.sidebar.subheader("Целевые веса (%)\")
        target_weights_input: Dict[str, float] = {}
        if tickers:
            cols = st.columns(2) # Убрал st.sidebar
            col_idx = 0
            assets_for_weighting = tickers + ['Cash']

            for asset in assets_for_weighting: # asset здесь - это тикер (SPY, GLD...) или 'Cash'
                session_key = f"weight_{asset}"
                # Убедимся, что ключ существует в состоянии (это должно делаться в callback)
                if session_key not in st.session_state:
                     # Эта логика должна быть в adjust_weights, но на всякий случай оставим fallback
                     num_assets_plus_cash = len(tickers) + 1
                     fallback_default = 100.0 / num_assets_plus_cash if num_assets_plus_cash > 0 else 0
                     st.session_state[session_key] = fallback_default

                with cols[col_idx % len(cols)]:
                    st.markdown(f"<small>Вес {asset}</small>", unsafe_allow_html=True)
                    # Убираем key и передаем value явно из session_state
                    weight = st.number_input(
                        label=f"Вес {asset}", # Label нужен для внутренней идентификации, но он скрыт
                        label_visibility="collapsed",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state[session_key], # Явно берем значение из состояния
                        step=1.0
                        # key=session_key # Убрали key, чтобы избежать конфликта
                    )
                    target_weights_input[asset] = weight / 100.0 # Собираем значение как и раньше
                col_idx += 1

            # Показываем сумму весов и предупреждение
            # Сумму считаем из ТЕКУЩИХ значений виджетов (target_weights_input), а не из session_state
            current_sum_weights = sum(target_weights_input.values()) * 100
            st.metric("Текущая сумма весов", f"{current_sum_weights:.1f}%") # Убрал st.sidebar
            if abs(current_sum_weights - 100.0) > 0.1:
                st.warning("Сумма весов должна быть равна 100%!") # Убрал st.sidebar
        else:
            st.text("Выберите активы для задания весов.") # Сообщение, если тикеры не выбраны

    # Секция Динамических Весов
    with st.sidebar.expander("Динамические Веса (Beta)", expanded=False):
        use_dynamic_weights = st.checkbox("Включить Inverse Volatility", value=False, help="Веса пересчитываются обратно пропорционально волатильности.")
        
        dynamic_window = 90
        if use_dynamic_weights:
            dynamic_window = st.slider(
                "Окно расчета (дней)",
                min_value=30,
                max_value=365,
                value=90,
                step=30,
                help="Период для расчета исторической волатильности."
            )
            st.info(f"Веса будут обновляться ежедневно на основе волатильности за последние {dynamic_window} дн.")

# Кнопка запуска (остается вне expander)
run_button = st.sidebar.button("Запустить Бэктест")

# --- Основная область для вывода результатов ---
st.divider()

if run_button:
    # 1. Валидация ввода (проверяем tickers, даты, и сумму из target_weights_input)
    errors = []
    if not tickers:
        errors.append("Выберите хотя бы один актив.")
    if start_date >= end_date:
        errors.append("Начальная дата должна быть раньше конечной даты.")
    # Используем веса из target_weights_input для финальной проверки суммы
    if tickers and abs(sum(target_weights_input.values()) * 100 - 100.0) > 0.1:
        errors.append("Сумма целевых весов должна быть равна 100%.")

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    # Нормализуем финальные веса из target_weights_input
    total_weight_input = sum(target_weights_input.values())
    if total_weight_input > 0:
        normalized_weights = {k: v / total_weight_input for k, v in target_weights_input.items()}
    else:
         normalized_weights = {} # Если сумма 0, то весов нет

    # 2. Выполнение расчетов
    price_data = None
    backtest_results = None
    drawdown_results = None # Добавляем переменную для данных просадок
    metrics = None

    try:
        with st.spinner('Загрузка исторических данных...'):
            price_data = core.load_price_data(
                tickers,
                start_date,
                end_date
            )

        if price_data is None or price_data.empty:
            st.error("Не удалось загрузить данные. Проверьте выбранные активы и период.")
            st.stop()

        # Проверка на наличие всех тикеров после загрузки (на всякий случай)
        if not all(ticker in price_data.columns for ticker in tickers):
             st.error(f"Не удалось загрузить данные для всех указанных активов. Проверьте: {tickers}")
             st.stop()

        with st.spinner('Запуск бэктеста...'):
            # Запускаем бэктест
            backtest_output = core.run_backtest(
                price_data=price_data,
                target_weights=normalized_weights,
                rebalance_freq=rebalance_freq_display,
                initial_capital=initial_capital,
                weight_deviation_threshold=price_change_threshold,
                deviation_type=deviation_type,
                use_dynamic_weights=use_dynamic_weights, # Передаем параметр
                dynamic_window_days=dynamic_window
            )
            if backtest_output is None:
                st.error("Ошибка при выполнении бэктеста.")
                st.stop()
            # Распаковываем ПЯТЬ результатов
            backtest_results, drawdown_results, rebalance_log, hover_weights_text, auxiliary_data = backtest_output

        # 2.5 Расчет метрик
        if backtest_results is None or drawdown_results is None:
            st.error("Ошибка: Результаты бэктеста или просадок отсутствуют после выполнения.")
            st.stop()

        with st.spinner('Расчет итоговых метрик...'):
            try:
                # Вычисляем безрисковую ставку в виде десятичной дроби
                rf_rate_decimal = rf_rate_percent / 100.0

                # Рассчитываем метрики, передавая лог ребалансировок
                # Передаем только результаты стоимостей в метрики
                metrics = core.calculate_metrics(backtest_results, rf_rate_decimal, rebalance_log)
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка при расчете метрик: {e}")
                st.exception(e)
                st.stop()

    except Exception as e:
        st.error(f"Произошла непредвиденная ошибка при загрузке данных или бэктеста: {e}") # Обновил сообщение об ошибке
        st.exception(e)
        st.stop()

    # 3. Отображение результатов
    # Оборачиваем каждую секцию в expander
    with st.expander("1. Нормализованные цены активов", expanded=True):
        prices_fig = plots.plot_normalized_prices(price_data, selected_ticker_map)
        if prices_fig:
            st.plotly_chart(prices_fig, width='stretch')
        else:
            st.warning("Не удалось построить график нормализованных цен.")

    with st.expander("2. Динамика стоимости портфелей", expanded=True):
        equity_fig = plots.plot_equity_curves(backtest_results, selected_ticker_map, hover_weights_text)
        if equity_fig:
            st.plotly_chart(equity_fig, width='stretch')
        else:
            st.warning("Не удалось построить график эквити.")

    with st.expander("3. Итоговые метрики", expanded=True):
        if metrics:
            metrics_df = pd.DataFrame(metrics).T

            # Определяем порядок стратегий
            # Восстанавливаем Комби в main_order
            main_order = ['Ребаланс (Календарь)', 'Ребаланс (Доля %)', 'Ребаланс (Комби)', 'B&H (Целевые веса)']
            individual_bh_order = sorted(
                [name for name in metrics_df.index if name.startswith('B&H (') and name != 'B&H (Целевые веса)'],
                key=lambda x: x.split('(')[1]
            )
            strategy_order = main_order + individual_bh_order
            # Переиндексируем и удаляем строки, где все значения NaN
            metrics_df = metrics_df.reindex([s for s in strategy_order if s in metrics_df.index]).dropna(how='all')

            # --- Добавляем переименование индекса для B&H ---
            new_index = []
            for idx_name in metrics_df.index:
                if idx_name.startswith('B&H (') and idx_name != 'B&H (Целевые веса)':
                    try:
                        ticker = idx_name.split('(')[1].split(')')[0] # Извлекаем тикер
                        full_name = COMMON_TICKERS.get(ticker, ticker) # Получаем полное имя
                        new_index.append(f"B&H ({ticker} - {full_name})") # Формируем новую строку
                    except IndexError:
                        new_index.append(idx_name) # Если парсинг не удался, оставляем как есть
                else:
                    new_index.append(idx_name)
            metrics_df.index = new_index
            # ------------------------------------------------

            if not metrics_df.empty:
                metrics_df_display = metrics_df.copy()

                # --- Убираем временный вывод ---
                # st.write("Отладочные данные для RF:")
                # st.dataframe(metrics_df_display[['End Value', 'Max Drawdown %', 'Max Drawdown Abs $', 'Peak Before MDD', 'Recovery Factor']], use_container_width=True)
                # ---------------------------------

                # Форматирование (основное)
                metrics_df_display['End Value'] = metrics_df['End Value'].map("${:,.0f}".format)
                metrics_df_display['CAGR'] = metrics_df['CAGR'].map("{:.2%}".format)
                metrics_df_display['Max Drawdown %'] = metrics_df['Max Drawdown %'].map("{:.2f}%".format)
                # Убираем отображение новых колонок из финальной таблицы
                metrics_df_display['Volatility'] = metrics_df['Volatility'].map("{:.2%}".format)
                metrics_df_display['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].map("{:.2f}".format)
                metrics_df_display['Sortino Ratio'] = metrics_df['Sortino Ratio'].apply(lambda x: "∞" if np.isinf(x) else f"{x:.2f}")
                metrics_df_display['Recovery Factor'] = metrics_df['Recovery Factor'].apply(lambda x: "∞" if np.isinf(x) else f"{x:.2f}")

                # Убираем колонки Start Value, Abs Drawdown, Peak из финального вывода
                columns_to_drop = ['Start Value', 'Max Drawdown Abs $', 'Peak Before MDD']
                metrics_df_display = metrics_df_display.drop(columns=[col for col in columns_to_drop if col in metrics_df_display.columns])

                # Форматирование Rebalance Count (ДО переименования)
                if 'Rebalance Count' in metrics_df_display.columns:
                     metrics_df_display['Rebalance Count'] = metrics_df['Rebalance Count'].map(lambda x: "{:.0f}".format(x) if not pd.isna(x) else "-")

                # Названия колонок (финальные) - используем rename для надежности
                rename_map = {
                    "End Value": "Конечная стоимость",
                    "CAGR": "CAGR",
                    "Max Drawdown %": "Просадка %",
                    "Volatility": "Волат-ть",
                    "Sharpe Ratio": "Шарп",
                    "Sortino Ratio": "Сортино",
                    "Recovery Factor": "Ф. Восст.",
                    "Rebalance Count": "Ребал."
                }
                metrics_df_display = metrics_df_display.rename(columns=rename_map)

                st.dataframe(metrics_df_display, width='stretch')
            else:
                 st.warning("Не удалось рассчитать метрики ни для одной стратегии.")
        else:
            st.warning("Не удалось рассчитать метрики.")

    # Секция 4: Просадки портфелей (новый пункт)
    with st.expander("4. Просадки портфелей", expanded=True):
        drawdown_fig = plots.plot_drawdown_curves(drawdown_results)
        if drawdown_fig:
            st.plotly_chart(drawdown_fig, width='stretch')
        else:
            st.warning("Не удалось построить график просадок.")

    # Секция 5: Матрица корреляций
    with st.expander("5. Матрица корреляций", expanded=True):
        if price_data is not None and len(tickers) >= 2: # Проверяем наличие данных и >= 2 активов
            corr_fig = plots.plot_correlation_heatmap(price_data, selected_ticker_map)
            if corr_fig:
                st.plotly_chart(corr_fig, width='stretch')
            else:
                st.warning("Не удалось построить матрицу корреляций.")
        else:
            st.warning("Для расчета корреляции необходимо выбрать минимум 2 актива.")

    # Секция 6: Фактическое Распределение (Дрейф) - Всегда отображается
    with st.expander("6. Фактическое Распределение (Дрейф)", expanded=True):
        st.caption("Данные графики показывают реальное изменение долей активов внутри дня. Вы можете увидеть, как меняются доли из-за движения цен и как они возвращаются к целевым значениям при ребалансировке.")
        
        # Получаем данные
        aw_comb = auxiliary_data.get('actual_weights_combined')
        aw_cal = auxiliary_data.get('actual_weights_calendar')
        aw_band = auxiliary_data.get('actual_weights_band')
        
        # Extract rebalance dates from logs
        # rebalance_log имеет структуру: {'calendar': [...], 'weight_band': [...], 'combined': [...]}
        # Каждый элемент списка - кортеж (date, type)
        dates_comb = [x[0] for x in rebalance_log.get('combined', [])]
        dates_cal = [x[0] for x in rebalance_log.get('calendar', [])]
        dates_band = [x[0] for x in rebalance_log.get('weight_band', [])]
        
        tab_comb, tab_cal, tab_band = st.tabs(["Комбинированная (Combined)", "Календарная (Calendar)", "По Отклонению (Threshold)"])
        
        with tab_comb:
             if aw_comb is not None and not aw_comb.empty:
                fig_comb = plots.plot_effective_weights(aw_comb, selected_ticker_map, rebalance_dates=dates_comb)
                if fig_comb:
                     st.plotly_chart(fig_comb, width='stretch', key="aw_comb")
                else:
                     st.warning("Нет данных.")
             else:
                 st.warning("Не удалось рассчитать фактические веса для комбинированной стратегии.")

        with tab_cal:
             if aw_cal is not None and not aw_cal.empty:
                fig_cal = plots.plot_effective_weights(aw_cal, selected_ticker_map, rebalance_dates=dates_cal)
                if fig_cal:
                     st.plotly_chart(fig_cal, width='stretch', key="aw_cal")
                else:
                     st.warning("Нет данных.")
             else:
                 st.warning("Не удалось рассчитать фактические веса для календарной стратегии.")

        with tab_band:
             if aw_band is not None and not aw_band.empty:
                fig_band = plots.plot_effective_weights(aw_band, selected_ticker_map, rebalance_dates=dates_band)
                if fig_band:
                     st.plotly_chart(fig_band, width='stretch', key="aw_band")
                else:
                     st.warning("Нет данных.")
             else:
                 st.warning("Не удалось рассчитать фактические веса для стратегии по отклонению.")

    # Секция 7: Распределение Капитала ($)
    with st.expander("7. Распределение Капитала ($)", expanded=True):
        st.caption("Данные графики показывают абсолютную стоимость каждого актива в портфеле. Сумма всех областей соответствует общей стоимости портфеля (Equity Curve).")
        
        # Получаем данные о стоимости
        aa_comb = auxiliary_data.get('actual_alloc_combined')
        aa_cal = auxiliary_data.get('actual_alloc_calendar')
        aa_band = auxiliary_data.get('actual_alloc_band')
        
        dates_comb = [x[0] for x in rebalance_log.get('combined', [])]
        dates_cal = [x[0] for x in rebalance_log.get('calendar', [])]
        dates_band = [x[0] for x in rebalance_log.get('weight_band', [])]
        
        tab_aa_comb, tab_aa_cal, tab_aa_band = st.tabs(["Комбинированная (Combined)", "Календарная (Calendar)", "По Отклонению (Threshold)"])
        
        with tab_aa_comb:
             if aa_comb is not None and not aa_comb.empty:
                fig_aa_comb = plots.plot_absolute_allocation(aa_comb, selected_ticker_map, rebalance_dates=dates_comb)
                if fig_aa_comb:
                     st.plotly_chart(fig_aa_comb, width='stretch', key="aa_comb")
                else:
                     st.warning("Нет данных.")
             else:
                 st.warning("Не удалось рассчитать стоимость для комбинированной стратегии.")

        with tab_aa_cal:
             if aa_cal is not None and not aa_cal.empty:
                fig_aa_cal = plots.plot_absolute_allocation(aa_cal, selected_ticker_map, rebalance_dates=dates_cal)
                if fig_aa_cal:
                     st.plotly_chart(fig_aa_cal, width='stretch', key="aa_cal")
                else:
                     st.warning("Нет данных.")
             else:
                 st.warning("Не удалось рассчитать стоимость для календарной стратегии.")

        with tab_aa_band:
             if aa_band is not None and not aa_band.empty:
                fig_aa_band = plots.plot_absolute_allocation(aa_band, selected_ticker_map, rebalance_dates=dates_band)
                if fig_aa_band:
                     st.plotly_chart(fig_aa_band, width='stretch', key="aa_band")
                else:
                     st.warning("Нет данных.")
             else:
                 st.warning("Не удалось рассчитать стоимость для стратегии по отклонению.")

    # Секция 8: Анализ Inverse Volatility (Только если включено)
    if use_dynamic_weights:
        with st.expander("8. Анализ Стратегии Inverse Volatility", expanded=True):
            target_weights_df = auxiliary_data.get('dynamic_weights_df')
            vol_df = auxiliary_data.get('volatility_df')
            
            tab_tgt, tab_vol, tab_risk = st.tabs(["Целевые Веса (Target)", "Историческая Волатильность", "Вклад в Риск"])
            
            with tab_tgt:
                st.caption("Расчетные целевые веса, которые стратегия предлагает на каждый день (до применения ограничений ребалансировки).")
                if target_weights_df is not None and not target_weights_df.empty:
                    tgt_fig = plots.plot_effective_weights(target_weights_df, selected_ticker_map)
                    if tgt_fig:
                        st.plotly_chart(tgt_fig, width='stretch', key="iv_tgt")
                    else:
                        st.warning("Нет данных.")
            
            with tab_vol:
                st.caption(f"Скользящая волатильность за окно {dynamic_window} дней.")
                if vol_df is not None and not vol_df.empty:
                    vol_fig = plots.plot_rolling_volatility(vol_df, selected_ticker_map)
                    if vol_fig:
                        st.plotly_chart(vol_fig, width='stretch', key="iv_vol")
                    else:
                        st.warning("Нет данных.")

            with tab_risk:
                st.caption("Оценка вклада каждого актива в общий риск (Вес * Волатильность).")
                if vol_df is not None and not vol_df.empty and target_weights_df is not None:
                    risk_fig = plots.plot_weighted_risk_contribution(vol_df, target_weights_df, selected_ticker_map)
                    if risk_fig:
                        st.plotly_chart(risk_fig, width='stretch', key="iv_risk")
                    else:
                        st.warning("Нет данных.")

else:
    st.info("Настройте параметры в боковой панели и нажмите 'Запустить Бэктест'!") 