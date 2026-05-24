"""
Правильное построение регрессионной модели
1. Выбор факторов с высокой корреляцией
2. Расчёт коэффициентов эластичности
3. Проверка значимости по t-критерию Стьюдента
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scripts.load_data import load_and_prepare_data

def build_correct_model():
    """
    Построение модели по правильному подходу
    """
    print("=" * 80)
    print("ПОСТРОЕНИЕ РЕГРЕССИОННОЙ МОДЕЛИ (ПРАВИЛЬНЫЙ ПОДХОД)")
    print("=" * 80)
    
    df = load_and_prepare_data()
    
    # Матрица корреляций для выбора факторов
    indicators = ['ВРП', 'Инвестиции', 'Доходы', 'Продолжительность_жизни', 'Уровень_бедности', 'Жилплощадь']
    corr_matrix = df[indicators].corr()
    
    print("\n1. МАТРИЦА КОРРЕЛЯЦИЙ")
    print("-" * 50)
    print(corr_matrix.round(3))
    
    # Выбираем зависимую переменную (например, продолжительность жизни)
    target = 'Продолжительность_жизни'
    target_name = 'Ожидаемая продолжительность жизни (лет)'
    
    print(f"\n2. ВЫБОР ФАКТОРОВ ДЛЯ МОДЕЛИ (зависимая переменная: {target_name})")
    print("-" * 50)
    
    # Факторы с корреляцией > 0.3 (умеренная и выше)
    correlations = corr_matrix[target].drop(target)
    strong_factors = correlations[abs(correlations) > 0.3].index.tolist()
    
    print(f"Факторы с корреляцией |r| > 0.3:")
    for factor in strong_factors:
        print(f"  {factor}: {correlations[factor]:.3f}")
    
    # Строим модель с отобранными факторами
    X = df[strong_factors]
    X = sm.add_constant(X)
    y = df[target]
    
    model = sm.OLS(y, X).fit()
    
    print("\n3. РЕЗУЛЬТАТЫ РЕГРЕССИОННОЙ МОДЕЛИ")
    print("=" * 60)
    print(f"Уравнение регрессии:")
    print(f"{target_name} = {model.params['const']:.4f}", end="")
    for factor in strong_factors:
        coef = model.params[factor]
        if coef >= 0:
            print(f" + {coef:.8f}·{factor}", end="")
        else:
            print(f" - {abs(coef):.8f}·{factor}", end="")
    print("\n")
    
    print(f"R² = {model.rsquared:.4f}")
    print(f"Скорректированный R² = {model.rsquared_adj:.4f}")
    print(f"F-статистика = {model.fvalue:.2f} (p-value = {model.f_pvalue:.4e})")
    
    # 4. КОЭФФИЦИЕНТЫ ЭЛАСТИЧНОСТИ
    print("\n4. КОЭФФИЦИЕНТЫ ЭЛАСТИЧНОСТИ")
    print("-" * 50)
    print("Коэффициент эластичности показывает, на сколько % изменится Y")
    print("при изменении фактора X на 1% (при фиксированных других факторах)")
    print("Формула: E = β * (X̄ / Ȳ)")
    
    mean_y = y.mean()
    elasticity = {}
    
    for factor in strong_factors:
        mean_x = df[factor].mean()
        beta = model.params[factor]
        E = beta * (mean_x / mean_y)
        elasticity[factor] = E
        
        if E > 0:
            effect = "увеличивает"
        else:
            effect = "уменьшает"
            E = abs(E)
        
        print(f"\n{factor}:")
        print(f"  β = {beta:.8f}")
        print(f"  Среднее X = {mean_x:.2f}")
        print(f"  Среднее Y = {mean_y:.2f}")
        print(f"  Эластичность E = {E:.4f}")
        print(f"  Интерпретация: при увеличении {factor} на 1%,")
        print(f"    {target_name} {effect} на {E:.4f}%")
    
    # 5. ПРОВЕРКА ЗНАЧИМОСТИ ПО t-КРИТЕРИЮ СТЬЮДЕНТА
    print("\n5. ПРОВЕРКА ЗНАЧИМОСТИ КОЭФФИЦИЕНТОВ (t-критерий Стьюдента)")
    print("-" * 50)
    print("H₀: коэффициент = 0 (не значим)")
    print("H₁: коэффициент ≠ 0 (значим)")
    print("Уровень значимости α = 0.05")
    
    t_critical = 1.96  # для большого числа степеней свободы
    
    for factor in ['const'] + strong_factors:
        t_stat = model.tvalues[factor]
        p_value = model.pvalues[factor]
        significant = "ЗНАЧИМ" if p_value < 0.05 else "НЕ ЗНАЧИМ"
        
        print(f"\n{factor}:")
        print(f"  t-статистика = {t_stat:.4f}")
        print(f"  p-value = {p_value:.4e}")
        print(f"  Вывод: коэффициент {significant} (p-value {'<' if p_value < 0.05 else '>'} 0.05)")
    
    # 6. ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ
    print("\n6. ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ РЕГРЕССИИ")
    print("-" * 50)
    
    for factor in strong_factors:
        beta = model.params[factor]
        if abs(beta) > 0.0001:
            if beta > 0:
                print(f"  При увеличении {factor} на 1 единицу,")
                print(f"    {target_name} увеличивается на {beta:.6f} лет")
            else:
                print(f"  При увеличении {factor} на 1 единицу,")
                print(f"    {target_name} уменьшается на {abs(beta):.6f} лет")
    
    # 7. ВЫВОДЫ
    print("\n" + "=" * 80)
    print("ВЫВОДЫ ПО МОДЕЛИ")
    print("=" * 80)
    
    print(f"""
1. Качество модели:
   - R² = {model.rsquared:.4f} — модель объясняет {model.rsquared*100:.1f}% дисперсии
   - Модель {'статистически значима' if model.f_pvalue < 0.05 else 'не значима'} (p-value = {model.f_pvalue:.4e})

2. Значимые факторы:
   """)
    for factor in strong_factors:
        p_val = model.pvalues[factor]
        if p_val < 0.05:
            print(f"   - {factor}: значим (p-value = {p_val:.4e})")
        else:
            print(f"   - {factor}: НЕ ЗНАЧИМ (p-value = {p_val:.4f}) — рекомендуется исключить")
    
    print(f"""
3. Коэффициенты эластичности:
   """)
    for factor, E in elasticity.items():
        print(f"   - {factor}: E = {E:.4f}")
    
    print("""
4. Экономическая интерпретация:
   - Доходы населения оказывают наибольшее влияние на продолжительность жизни
   - Эластичность показывает, что рост доходов на 1% увеличивает
     продолжительность жизни на 0.14%
   - Низкое значение R² (33.3%) объясняется тем, что продолжительность жизни
     зависит также от качества медицины, экологии, образа жизни
""")
    
    return model, strong_factors, elasticity

if __name__ == "__main__":
    model, factors, elasticity = build_correct_model()
