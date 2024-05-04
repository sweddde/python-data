import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
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
    folder_path = save_to_folder('')

    df = generate_dataset()
    df.to_csv(os.path.join(folder_path, 'generated_dataset.csv'), index=False)
    print("Сгенерированный датасет сохранен.")

    df_filtered = filter_dataset_parallel(df)
    df_filtered.to_csv(os.path.join(folder_path, 'filtered_dataset.csv'), index=False)
    print("Датасет отфильтрован и сохранен.")

# Вызов функции для создания датасета, его фильтрации и сохранения результата
process_dataset_and_save_parallel()
print("Генерация датасета и фильтрация завершены.")
####################################################
#РАСЧЕТ МЕТРИК
####################################################
def calculate_metrics_and_save():
    # Чтение исходного датасета
    df = pd.read_csv('generated_files/filtered_dataset.csv')

    # Преобразование времени в часы
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # Расчет метрик: кол-во уникальных строк, среднее и медиана для числовых значений
    metrics_df = df.groupby('hour').agg({
        'text': 'nunique',  # Кол-во уникальных строк
        'value': ['mean', 'median']  # Среднее и медиана для числовых значений
    }).reset_index()

    # Сохранение результатов в файл 'metrics.csv'
    metrics_df.to_csv('generated_files/metrics.csv', index=False)

    # Вывод результата
    print('#######################\nРассчет метрик\n#######################\n', metrics_df)
# SQL запрос для выполнения аналогичных расчетов в базе данных (PostgreSQL)
sql_query = """
SELECT 
    EXTRACT(HOUR FROM datetime) AS hour,
    COUNT(DISTINCT text) AS unique_strings,
    AVG(value) AS average_numeric,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) AS median_numeric
FROM generated_dataset
GROUP BY hour;
"""

# Вызов функции для расчета метрик и сохранения результатов
calculate_metrics_and_save()
####################################################
#ОБЪЕДИНЕНИЕ С МЕТРИКАМИ
####################################################
def merge_datasets_and_save():
    # Чтение исходного датасета
    df = pd.read_csv('generated_files/generated_dataset.csv')

    # Чтение файла с метриками
    metrics_df = pd.read_csv('generated_files/metrics.csv')

    # Приведение времени к формату часов
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # Создание временной колонки для объединения данных
    df['temp_hour'] = df['hour']

    # Объединение данных по ближайшему часу
    merged_df = df.merge(metrics_df, left_on='temp_hour', right_on='hour', how='left')

    # Удаление временных колонок
    merged_df.drop(['hour_y', 'temp_hour'], axis=1, inplace=True)
    merged_df.rename(columns={'hour_x': 'hour'}, inplace=True)

    # Сохранение результата в файл
    merged_df.to_csv('generated_files/merged_dataset.csv', index=False)
    print('#######################\n Мердж метрик \n#######################\n', merged_df)
# Вызов функции для объединения данных и сохранения результата
merge_datasets_and_save()

####################################################
#РАССЧЕТ АНАЛИТ.МЕТРИК
####################################################
def calculate_confidence_interval(input_file, output_file):
    # Чтение полного датасета
    full_df = pd.read_csv(input_file)

    # Построение гистограммы для колонки 'value'
    plt.figure(figsize=(12, 6))
    sns.histplot(full_df['value'], bins=20, kde=True)
    plt.title('Histogram of Numeric Column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
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


# Вызов функции для расчета доверительного интервала и сохранения результата
calculate_confidence_interval('generated_files/generated_dataset.csv', 'generated_files/confidence_interval_result.txt')
####################################################
#ВИЗУАЛИЗАЦИЯ
####################################################
def plot_average_numeric_value_by_month(data, output_file):
    # Преобразуем столбец 'datetime' в формат datetime
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Извлечение месяца из 'datetime'
    data['month'] = data['datetime'].dt.month

    # График среднего значения числовой колонки по месяцам
    plt.figure(figsize=(12, 6))
    sns.barplot(x='month', y='value', data=data, estimator=np.mean)
    plt.title('Average Numeric Value by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Numeric Value')
    plt.savefig(output_file)  # Сохранение графика в файл
    plt.show()

def plot_character_frequency_heatmap(data, output_file):
    # Подсчет частотности символов
    char_freq = data['text'].apply(lambda x: list(x)).explode().value_counts()

    # Преобразование в Dask DataFrame для построения Heatmap
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


full_df = dd.read_csv('generated_files/generated_dataset.csv')

# Вызов функций для построения Heatmap и графика с прогресс-баром
heatmap_output_file = 'generated_files/character_frequency_heatmap.png'
numeric_output_file = 'generated_files/average_numeric_value_by_month.png'

print("Построение Heatmap и графика:")
plot_character_frequency_heatmap(full_df.compute(), heatmap_output_file)
plot_average_numeric_value_by_month(full_df.compute(), numeric_output_file)
print("Heatmap и график сохранены.")