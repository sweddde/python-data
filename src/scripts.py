import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

####################################################
#ГЕНЕРАЦИЯ ДАТАСЕТА
####################################################
def save_to_folder(file_name, folder_name='generated_files'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, file_name)
    return file_path
# Создаем рандомный датасет из 1млн записей
def generate_dataset(num_rows=1000000, pct_duplicates=0.1):
    data = {
        'id': np.arange(num_rows),
        'datetime': pd.date_range(start='2020-01-01', end='2023-12-31', periods=num_rows),
        'value': np.random.rand(num_rows) * 1000,
        'text': [''.join(np.random.choice(['z', 'x', 'c', 'a', 'w', 'g', '1', '99', '0000'], size=np.random.randint(5, 20))) for _ in range(num_rows)],
    }
# Добавим дублей
    duplicate_ids = np.random.choice(data['id'], size=int(num_rows * pct_duplicates), replace=False)
    for i in duplicate_ids:
        data['text'][i] = ''.join(np.random.choice(['z', 'x', 'c', 'a', 'w', 'g', '1', '99', '0000'], size=np.random.randint(5, 20)))

    return pd.DataFrame(data)
####################################################
#ФИЛЬТРАЦИЯ ДАТАСЕТА
####################################################
def filter_dataset(df):
    ddf = dd.from_pandas(df, npartitions=4)  # Создание Dask DataFrame
    ddf = ddf.dropna(how='any')  # Удаление пустых/na строк
    ddf = ddf.drop_duplicates()  # Удаление дубликатов
    ddf['text'] = ddf['text'].apply(lambda text: '' if not any(char.isdigit() for char in str(text)) else text, meta=('text', 'str'))  # Замена строк без цифр на пустые строки
    ddf['datetime'] = dd.to_datetime(ddf['datetime'])
    ddf = ddf[~((ddf['datetime'].dt.hour >= 1) & (ddf['datetime'].dt.hour < 3))]  # Удаление записей в промежутке с 1 до 3 часов ночи

    df_filtered = ddf.compute()
    return df_filtered

def filter_partition(partition):
    # Функция фильтрации для применения к каждому разделу данных
    filtered_partition = partition.dropna(subset=['datetime'])
    filtered_partition = filtered_partition.drop_duplicates()
    filtered_partition['text'] = filtered_partition['text'].apply(lambda text: '' if not any(char.isdigit() for char in str(text)) else text)
    filtered_partition['datetime'] = dd.to_datetime(filtered_partition['datetime'])
    filtered_partition = filtered_partition[~((filtered_partition['datetime'].dt.hour >= 1) & (filtered_partition['datetime'].dt.hour < 3))]
    return filtered_partition

def filter_dataset_parallel(df):
    ddf = dd.from_pandas(df, npartitions=4)  # Создание Dask DataFrame с заданным количеством разделов

    # Применение функции filter_partition к каждому разделу параллельно
    filtered_ddf = ddf.map_partitions(filter_partition)

    # Выполнение параллельной обработки
    df_filtered = filtered_ddf.compute()
    return df_filtered


def process_dataset_and_save_parallel():
    print("Генерация датасета и фильтрация:")
    folder_path = save_to_folder('', folder_name='generated_files')

    df = generate_dataset()
    df.to_csv(os.path.join(folder_path, 'generated_dataset.csv'), index=False)
    print("Сгенерированный датасет сохранен.")

    df_filtered = filter_dataset_parallel(df)
    df_filtered.to_csv(os.path.join(folder_path, 'filtered_dataset.csv'), index=False)
    print("Датасет отфильтрован и сохранен.")
####################################################
#РАСЧЕТ МЕТРИК
####################################################

def calculate_metrics_and_save():
    # Чтение фильтрованного датасета
    df = pd.read_csv('generated_files/filtered_dataset.csv')

    # Преобразование столбца datetime в тип данных datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Группировка данных по часам и выполнение агрегации
    metrics_df = df.groupby(df['datetime'].dt.floor('h')).agg(
        unique_strings=('text', 'nunique'),  # количество уникальных строк
        average_numeric=('value', 'mean'),  # среднее значение числового столбца
        median_numeric=('value', 'median')  # медиана числового столбца
    )

    # Создание столбца 'datetime' на основе индекса времени
    metrics_df['datetime'] = metrics_df.index

    # Установка столбца 'datetime' в качестве индекса
    metrics_df.set_index('datetime', inplace=True)

    # Сохранение результатов в файл 'metrics.csv'
    metrics_df.to_csv('generated_files/metrics.csv')

    # Вывод результата
    print('#######################\nРассчет метрик\n#######################\n', metrics_df)

# SQL запрос для выполнения аналогичных расчетов в базе данных (PostgreSQL)
#sql_query = """
#SELECT
#    DATE_TRUNC('hour', datetime) AS hour,
#    COUNT(DISTINCT text) AS unique_strings,
#    AVG(value) AS average_numeric,
#    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) AS median_numeric
#FROM generated_dataset
#GROUP BY DATE_TRUNC('hour', datetime);
#"""

# Вызов функции для расчета метрик и сохранения результатов
####################################################
# ОБЪЕДИНЕНИЕ С МЕТРИКАМИ
####################################################
def calculate_metrics_and_merge():
    # Чтение сгенерированного датасета
    df = pd.read_csv('generated_files/generated_dataset.csv')

    # Преобразование столбца datetime в тип данных datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Чтение метрик
    metrics = pd.read_csv('generated_files/metrics.csv')

    # Преобразование столбца datetime к типу datetime
    metrics['datetime'] = pd.to_datetime(metrics['datetime'])

    # Объединение данных по ближайшему времени
    merged_df = pd.merge_asof(df.sort_values('datetime'), metrics.sort_values('datetime'), on='datetime', direction='nearest')

    # Сохранение результата в файл
    merged_df.to_csv('generated_files/merged_dataset.csv', index=False)

    # Вывод результата
    print('#######################\n Мердж метрик \n#######################\n', merged_df)
####################################################
# РАССЧЕТ АНАЛИТ.МЕТРИК
####################################################
def calculate_confidence_interval_pre(input_file, output_file, hist_output_file):
    # Чтение полного датасета
    full_df = pd.read_csv(input_file)

    # Создание гистограммы
    plt.figure(figsize=(12, 6))
    sns.histplot(full_df['value'], bins=10000, kde=True)
    plt.title('Histogram of Numeric Column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Сохранение гистограммы в файл
    plt.savefig(hist_output_file)

    # Отображение гистограммы
    plt.show()

    # Рассчет 95% доверительного интервала
    confidence_level = 0.95
    data = full_df['value']

    # Выбор методики расчета доверительного интервала
    mean = np.mean(data)
    std_err = stats.sem(data)
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = z_score * std_err
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    # Запись результата в файл
    with open(output_file, 'w') as file:
        file.write(f"95% Confidence Interval: {confidence_interval}")
    print(f'Доверительный интервал: {confidence_interval}')
def calculate_confidence_interval():
    calculate_confidence_interval_pre('generated_files/generated_dataset.csv',
                                      'generated_files/confidence_interval_result.txt',
                                      'generated_files/histogram.png')
####################################################
# ВИЗУАЛИЗАЦИЯ
####################################################
def plot_average_numeric_value_by_month(data, output_file):
    # Преобразуем столбец 'datetime' в формат datetime
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Извлечение месяца из 'datetime'
    data['month'] = data['datetime'].dt.month

    # Рассчет среднего значения числовой колонки по месяцам
    monthly_avg = data.groupby('month')['value'].mean()

    # График среднего значения числовой колонки по месяцам
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', color='b', linestyle='-')
    plt.xticks(range(1, 13), ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
    plt.title('Average Numeric Value by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Numeric Value')
    plt.grid(True)
    plt.savefig(output_file)  # Сохранение графика в файл
    plt.show()
def plot_character_frequency_heatmap(data, output_file):
    # Подсчет частотности символов
    char_freq = data['text'].apply(lambda x: list(x)).explode().value_counts()

    # Преобразование в DataFrame для построения Heatmap
    char_freq_df = char_freq.reset_index()
    char_freq_df.columns = ['character', 'frequency']

    # Построение Heatmap
    plt.figure(figsize=(12, 6))
    heatmap_data = char_freq_df.pivot_table(index='character', columns='frequency', aggfunc=len, fill_value=0)
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt='g')
    plt.title('Character Frequency Heatmap')
    plt.xlabel('Frequency')
    plt.ylabel('Character')
    plt.savefig(output_file)  # Сохранение Heatmap в файл
    plt.show()


# Чтение полного датасета с помощью Dask
full_df = dd.read_csv('generated_files/generated_dataset.csv')

# Вызов функций для построения Heatmap и графика
heatmap_output_file = 'generated_files/character_frequency_heatmap.png'
numeric_output_file = 'generated_files/average_numeric_value_by_month.png'
# Объединим две функции визуализации в одну для удобства
def visualize_data():
    plot_average_numeric_value_by_month(full_df.compute(), numeric_output_file)
    plot_character_frequency_heatmap(full_df.compute(), heatmap_output_file)

#объединим все функции в одну
def run_all_operations():
    process_dataset_and_save_parallel()
    calculate_metrics_and_save()
    calculate_metrics_and_merge()
    calculate_confidence_interval()
    visualize_data()