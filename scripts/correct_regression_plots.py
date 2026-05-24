"""
Правильное построение регрессионных моделей:
- Одна зависимая переменная (Продолжительность жизни)
- 1-2 фактора с высокой корреляцией
- Все типы регрессии на одном графике (точки + линии тренда)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scripts.load_data import load_and_prepare_data

def plot_regression_types(X, y, factor_name, target_name, output_filename):
    """
    Построение одного графика со всеми типами регрессии
    """
    # Очистка от NaN
    mask = ~(np.isnan(X) | np.isnan(y))
    X_clean = X[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    # Сортируем для плавных линий
    sort_idx = np.argsort(X_clean[:, 0])
    X_sorted = X_clean[sort_idx]
    
    plt.figure(figsize=(12, 8))
    
    # Исходные данные (точки)
    plt.scatter(X_clean, y_clean, alpha=0.4, color='gray', label='Данные', s=30)
    
    # 1. Линейная регрессия
    model_lin = LinearRegression()
    model_lin.fit(X_clean, y_clean)
    y_pred_lin = model_lin.predict(X_sorted)
    r2_lin = r2_score(y_clean, model_lin.predict(X_clean))
    plt.plot(X_sorted, y_pred_lin, 'b-', linewidth=2, label=f'Линейная (R²={r2_lin:.4f})')
    
    # 2. Экспоненциальная регрессия
    y_log = np.log(y_clean)
    model_exp = LinearRegression()
    model_exp.fit(X_clean, y_log)
    y_pred_exp = np.exp(model_exp.predict(X_sorted))
    r2_exp = r2_score(y_clean, np.exp(model_exp.predict(X_clean)))
    plt.plot(X_sorted, y_pred_exp, 'g-', linewidth=2, label=f'Экспоненциальная (R²={r2_exp:.4f})')
    
    # 3. Логарифмическая регрессия (только для положительных X)
    if np.min(X_clean) > 0:
        X_log = np.log(X_clean)
        model_log = LinearRegression()
        model_log.fit(X_log, y_clean)
        X_sorted_log = np.log(X_sorted)
        y_pred_log = model_log.predict(X_sorted_log)
        r2_log = r2_score(y_clean, model_log.predict(X_log))
        plt.plot(X_sorted, y_pred_log, 'orange', linewidth=2, label=f'Логарифмическая (R²={r2_log:.4f})')
    else:
        r2_log = None
    
    # 4. Полиномиальная 2-й степени
    poly2 = PolynomialFeatures(degree=2)
    X_poly2 = poly2.fit_transform(X_clean)
    model_poly2 = LinearRegression()
    model_poly2.fit(X_poly2, y_clean)
    X_sorted_poly2 = poly2.fit_transform(X_sorted)
    y_pred_poly2 = model_poly2.predict(X_sorted_poly2)
    r2_poly2 = r2_score(y_clean, model_poly2.predict(X_poly2))
    plt.plot(X_sorted, y_pred_poly2, 'purple', linewidth=2, label=f'Полином 2-й степени (R²={r2_poly2:.4f})')
    
    # 5. Полиномиальная 3-й степени
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X_clean)
    model_poly3 = LinearRegression()
    model_poly3.fit(X_poly3, y_clean)
    X_sorted_poly3 = poly3.fit_transform(X_sorted)
    y_pred_poly3 = model_poly3.predict(X_sorted_poly3)
    r2_poly3 = r2_score(y_clean, model_poly3.predict(X_poly3))
    plt.plot(X_sorted, y_pred_poly3, 'red', linewidth=2, label=f'Полином 3-й степени (R²={r2_poly3:.4f})')
    
    # Оформление
    plt.xlabel(factor_name, fontsize=12)
    plt.ylabel(target_name, fontsize=12)
    plt.title(f'{target_name} от {factor_name}\nСравнение различных типов регрессии', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    # Вывод результатов
    print(f"\n{factor_name} → {target_name}:")
    print(f"  Линейная: R² = {r2_lin:.4f}")
    print(f"  Экспоненциальная: R² = {r2_exp:.4f}")
    if r2_log:
        print(f"  Логарифмическая: R² = {r2_log:.4f}")
    print(f"  Полином 2-й степени: R² = {r2_poly2:.4f}")
    print(f"  Полином 3-й степени: R² = {r2_poly3:.4f}")
    
    # Лучшая модель
    r2_values = [r2_lin, r2_exp, r2_poly2, r2_poly3]
    if r2_log:
        r2_values.append(r2_log)
    best_r2 = max(r2_values)
    print(f"  ЛУЧШАЯ: R² = {best_r2:.4f}")
    
    return best_r2

def main():
    print("=" * 80)
    print("ПОСТРОЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ ДЛЯ ПРОДОЛЖИТЕЛЬНОСТИ ЖИЗНИ")
    print("=" * 80)
    
    df = load_and_prepare_data()
    
    # Названия
    names = {
        'ВРП': 'ВРП на душу (руб.)',
        'Инвестиции': 'Инвестиции на душу (руб.)',
        'Доходы': 'Доходы населения (руб.)'
    }
    
    target = 'Продолжительность_жизни'
    target_name = 'Ожидаемая продолжительность жизни (лет)'
    
    print(f"\nЗависимая переменная: {target_name}")
    print(f"\nПостроение графиков для каждого фактора:")
    print("-" * 50)
    
    results = {}
    
    for factor in ['Доходы', 'ВРП', 'Инвестиции']:
        print(f"\nФактор: {names[factor]}")
        X = df[factor].values
        y = df[target].values
        filename = f'results/figures/regression_{target}_from_{factor}.png'
        best_r2 = plot_regression_types(X, y, names[factor], target_name, filename)
        results[factor] = best_r2
    
    # Вывод итогов
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print("\nЛучшая модель для каждого фактора (по R²):")
    for factor, r2 in results.items():
        print(f"  {factor}: R² = {r2:.4f}")

if __name__ == "__main__":
    main()
