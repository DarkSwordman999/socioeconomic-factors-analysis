import pandas as pd
import numpy as np
from scripts.load_data import load_and_prepare_data

def calculate_statistics(df):
    """Расчёт основных статистических показателей"""
    indicators = ['ВРП', 'Инвестиции', 'Доходы', 'Уровень_бедности', 'Продолжительность_жизни', 'Жилплощадь']
    names = ['ВРП на душу (руб.)', 'Инвестиции на душу (руб.)', 'Среднедушевые доходы (руб.)',
             'Уровень бедности (%)', 'Ожидаемая продолжительность жизни (лет)', 'Жилплощадь на человека (кв.м)']
    
    results = []
    
    for col, name in zip(indicators, names):
        data = df[col]
        
        mean_val = data.mean()
        median_val = data.median()
        variance_val = data.var(ddof=1)
        std_val = data.std(ddof=1)
        range_val = data.max() - data.min()
        cv_val = (std_val / mean_val) * 100
        min_val = data.min()
        max_val = data.max()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        skew_val = data.skew()
        kurt_val = data.kurtosis()
        
        results.append({
            'Показатель': name,
            'Среднее': mean_val,
            'Медиана': median_val,
            'Дисперсия': variance_val,
            'СКО': std_val,
            'Размах': range_val,
            'Коэф. вариации (%)': cv_val,
            'Q1': q1,
            'Q3': q3,
            'Мин': min_val,
            'Макс': max_val,
            'Асимметрия': skew_val,
            'Эксцесс': kurt_val
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_and_prepare_data()
    results_df = calculate_statistics(df)
    results_df.to_csv('results/tables/descriptive_statistics_full.csv', index=False)
    print("Результаты сохранены в results/tables/descriptive_statistics_full.csv")
