import pandas as pd
import yfinance as yf

def _internal_load(tickers, start_date, end_date, loading_mode=None):
    """
    Простая функция загрузки данных через yfinance без прокси.
    """
    try:
        # Простая загрузка через yfinance
        data = yf.download(tickers, start=start_date, end=end_date)

        if data.empty:
            return None

        # Извлекаем 'Adj Close' если есть, иначе 'Close'
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        else:
            price_data = data['Close']

        # Удаляем NaN
        price_data = price_data.dropna()

        if price_data.empty:
            return None

        # Если один тикер, преобразуем в DataFrame
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])

        # Добавляем столбец Cash
        prices_cleaned = price_data.copy()
        prices_cleaned['Cash'] = 1.0

        prices_final = prices_cleaned.sort_index()

        return prices_final

    except Exception as e:
        print(f"Error loading data: {e}")
        return None
