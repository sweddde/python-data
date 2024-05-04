from src.scripts import *

def print_menu():
    print("Выберите действие:")
    print("1. Сгенерировать датасет")
    print("2. Отфильтровать датасет")
    print("3. Рассчитать метрики")
    print("4. Объединить данные с метриками")
    print("5. Рассчитать доверительный интервал")
    print("6. Визуализация данных")
    print("7. Выполнить все действия сразу")
    print("8. Выход")

def main():
    while True:
        print_menu()
        choice = input("Введите номер действия: ")

        if choice == '1':
            process_dataset_and_save_parallel()
        elif choice == '2':
            df = pd.read_csv('generated_files/generated_dataset.csv')
            df_filtered = filter_dataset(df)
            df_filtered.to_csv('generated_files/filtered_dataset.csv', index=False)
            print("Датасет отфильтрован и сохранен в файл 'generated_files/filtered_dataset.csv'")
        elif choice == '3':
            calculate_metrics_and_save()
        elif choice == '4':
            merge_datasets_and_save()
        elif choice == '5':
            calculate_confidence_interval('generated_files/generated_dataset.csv', 'generated_files/confidence_interval_result.txt')
        elif choice == '6':
            full_df = pd.read_csv('generated_files/generated_dataset.csv')
            plot_average_numeric_value_by_month(full_df, 'generated_files/average_numeric_value_by_month.png')
            plot_character_frequency_heatmap(full_df, 'generated_files/character_frequency_heatmap.png')
        elif choice == '7':
            process_dataset_and_save_parallel()
            calculate_metrics_and_save()
            merge_datasets_and_save()
            calculate_confidence_interval('generated_files/generated_dataset.csv', 'generated_files/confidence_interval_result.txt')
            full_df = pd.read_csv('generated_files/generated_dataset.csv')
            plot_average_numeric_value_by_month(full_df, 'generated_files/average_numeric_value_by_month.png')
            plot_character_frequency_heatmap(full_df, 'generated_files/character_frequency_heatmap.png')
            print("Все действия выполнены.")
        elif choice == '8':
            print("Программа завершена.")
            break
        else:
            print("Некорректный выбор. Пожалуйста, введите номер действия.")

if __name__ == "__main__":
    main()
