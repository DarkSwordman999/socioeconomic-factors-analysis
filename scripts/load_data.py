import pandas as pd
import numpy as np

def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    df = pd.read_csv('data/processed/final_dataset.csv')
    
    # Приведение к числовым типам
    for col in ['ВРП', 'Инвестиции', 'Доходы', 'Уровень_бедности', 'Продолжительность_жизни', 'Жилплощадь']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(f"Данные загружены: {len(df)} записей")
    print(f"Период: {df['Год'].min()} - {df['Год'].max()}")
    print(f"Регионов: {df['Регион'].nunique()}")
