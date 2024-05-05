from src.scripts import process_dataset_and_save_parallel, calculate_metrics_and_save, calculate_metrics_and_merge, \
    calculate_confidence_interval, visualize_data
import dask.dataframe as dd


if __name__ == "__main__":
    process_dataset_and_save_parallel()
    calculate_metrics_and_save()
    calculate_metrics_and_merge()
    calculate_confidence_interval()
    full_df = dd.read_csv('generated_files/generated_dataset.csv')
    visualize_data(full_df)