import pandas as pd
from scripts.load_data import load_and_prepare_data

def create_report_table(df):
    indicators = ['ВРП', 'Инвестиции', 'Доходы', 'Уровень_бедности', 'Продолжительность_жизни', 'Жилплощадь']
    names = ['ВРП на душу (руб.)', 'Инвестиции на душу (руб.)', 'Среднедушевые доходы (руб.)',
             'Уровень бедности (%)', 'Ожидаемая продолжительность жизни (лет)', 'Жилплощадь на человека (кв.м)']
    
    report_table = pd.DataFrame({
        'Показатель': names,
        'Среднее': [f"{df[col].mean():.2f}" for col in indicators],
        'Медиана': [f"{df[col].median():.2f}" for col in indicators],
        'СКО': [f"{df[col].std():.2f}" for col in indicators],
        'Коэф. вариации (%)': [f"{(df[col].std()/df[col].mean()*100):.2f}" for col in indicators],
        'Асимметрия': [f"{df[col].skew():.3f}" for col in indicators],
        'Эксцесс': [f"{df[col].kurtosis():.3f}" for col in indicators],
        'Мин': [f"{df[col].min():.2f}" for col in indicators],
        'Макс': [f"{df[col].max():.2f}" for col in indicators]
    })
    return report_table

if __name__ == "__main__":
    df = load_and_prepare_data()
    report_table = create_report_table(df)
    report_table.to_csv('results/tables/report_table.csv', index=False)
    print("Итоговая таблица сохранена в results/tables/report_table.csv")
    print("\n" + report_table.to_string(index=False))
