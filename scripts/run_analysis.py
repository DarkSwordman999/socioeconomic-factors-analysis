import pandas as pd
from scripts.load_data import load_and_prepare_data
from scripts.descriptive_stats import calculate_statistics

def main():
    # Загрузка данных
    df = load_and_prepare_data()
    
    print("=" * 80)
    print("ДЕСКРИПТИВНАЯ СТАТИСТИКА ПО РЕГИОНАМ РФ")
    print("Источник: Росстат, сборник 'Регионы России'")
    print("Период: 2000-2023 гг.")
    print(f"Количество наблюдений: {len(df)}")
    print("=" * 80)
    
    # 1. СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ
    print("\n1. ОСНОВНЫЕ СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ")
    print("-" * 80)
    
    results_df = calculate_statistics(df)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(results_df.to_string(index=False))
    
    # 2. ОЦЕНКА ОДНОРОДНОСТИ СОВОКУПНОСТИ
    print("\n" + "=" * 80)
    print("2. ОЦЕНКА ОДНОРОДНОСТИ СОВОКУПНОСТИ (коэффициент вариации)")
    print("-" * 80)
    print("По шкале: <33% - совокупность однородна, >33% - неоднородна")
    print("-" * 50)
    
    indicators = ['ВРП', 'Инвестиции', 'Доходы', 'Уровень_бедности', 'Продолжительность_жизни', 'Жилплощадь']
    names = ['ВРП на душу (руб.)', 'Инвестиции на душу (руб.)', 'Среднедушевые доходы (руб.)',
             'Уровень бедности (%)', 'Ожидаемая продолжительность жизни (лет)', 'Жилплощадь на человека (кв.м)']
    
    for col, name in zip(indicators, names):
        cv = results_df.loc[results_df['Показатель'] == name, 'Коэф. вариации (%)'].values[0]
        homogeneity = "однородна" if cv < 33 else "неоднородна"
        print(f"{name:<40} CV = {cv:.2f}% -> совокупность {homogeneity}")
    
    # 3. ХАРАКТЕРИСТИКИ ФОРМЫ РАСПРЕДЕЛЕНИЯ
    print("\n" + "=" * 80)
    print("3. ХАРАКТЕРИСТИКИ ФОРМЫ РАСПРЕДЕЛЕНИЯ")
    print("-" * 80)
    
    for col, name in zip(indicators, names):
        skew = results_df.loc[results_df['Показатель'] == name, 'Асимметрия'].values[0]
        kurt = results_df.loc[results_df['Показатель'] == name, 'Эксцесс'].values[0]
        
        if skew > 0:
            skew_type = "правосторонняя"
        elif skew < 0:
            skew_type = "левосторонняя"
        else:
            skew_type = "симметричное"
        
        if kurt > 0:
            kurt_type = "островершинное"
        elif kurt < 0:
            kurt_type = "плосковершинное"
        else:
            kurt_type = "нормальное"
        
        print(f"{name:<40} асимметрия = {skew:.3f} ({skew_type}), эксцесс = {kurt:.3f} ({kurt_type})")
    
    # Сохранение результатов
    results_df.to_csv('results/tables/descriptive_statistics_full.csv', index=False)
    print("\n" + "=" * 80)
    print("Результаты сохранены в results/tables/descriptive_statistics_full.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
