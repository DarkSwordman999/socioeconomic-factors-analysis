"""
Построение всех типов регрессии для каждой пары переменных
На одном полотне — 5 графиков (линейная, экспоненциальная, логарифмическая, полином 2, полином 3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scripts.load_data import load_and_prepare_data

def plot_all_regression_types(X, y, x_name, y_name, output_filename):
    """
    Создание одного полотна с 5 графиками (по одному на каждый тип регрессии)
    """
    # Очистка от NaN
    mask = ~(np.isnan(X) | np.isnan(y))
    X_clean = X[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    # Сортируем для плавных линий
    sort_idx = np.argsort(X_clean[:, 0])
    X_sorted = X_clean[sort_idx]
    y_sorted = y_clean[sort_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. ЛИНЕЙНАЯ РЕГРЕССИЯ
    model_lin = LinearRegression()
    model_lin.fit(X_clean, y_clean)
    y_pred_lin = model_lin.predict(X_sorted)
    r2_lin = r2_score(y_clean, model_lin.predict(X_clean))
    
    axes[0].scatter(X_clean, y_clean, alpha=0.3, color='gray', s=10)
    axes[0].plot(X_sorted, y_pred_lin, 'b-', linewidth=2, label=f'Линейная (R²={r2_lin:.4f})')
    axes[0].set_title(f'Линейная регрессия\nR² = {r2_lin:.4f}')
    axes[0].set_xlabel(x_name)
    axes[0].set_ylabel(y_name)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. ЭКСПОНЕНЦИАЛЬНАЯ РЕГРЕССИЯ (y = a * exp(b*x))
    y_log = np.log(y_clean)
    model_exp = LinearRegression()
    model_exp.fit(X_clean, y_log)
    y_pred_exp = np.exp(model_exp.predict(X_sorted))
    r2_exp = r2_score(y_clean, np.exp(model_exp.predict(X_clean)))
    
    axes[1].scatter(X_clean, y_clean, alpha=0.3, color='gray', s=10)
    axes[1].plot(X_sorted, y_pred_exp, 'g-', linewidth=2, label=f'Экспоненциальная (R²={r2_exp:.4f})')
    axes[1].set_title(f'Экспоненциальная регрессия\nR² = {r2_exp:.4f}')
    axes[1].set_xlabel(x_name)
    axes[1].set_ylabel(y_name)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. ЛОГАРИФМИЧЕСКАЯ РЕГРЕССИЯ (y = a + b*ln(x))
    X_log = np.log(X_clean)
    model_log = LinearRegression()
    model_log.fit(X_log, y_clean)
    X_sorted_log = np.log(X_sorted)
    y_pred_log = model_log.predict(X_sorted_log)
    r2_log = r2_score(y_clean, model_log.predict(X_log))
    
    axes[2].scatter(X_clean, y_clean, alpha=0.3, color='gray', s=10)
    axes[2].plot(X_sorted, y_pred_log, 'orange', linewidth=2, label=f'Логарифмическая (R²={r2_log:.4f})')
    axes[2].set_title(f'Логарифмическая регрессия\nR² = {r2_log:.4f}')
    axes[2].set_xlabel(x_name)
    axes[2].set_ylabel(y_name)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. ПОЛИНОМИАЛЬНАЯ 2-Й СТЕПЕНИ
    poly2 = PolynomialFeatures(degree=2)
    X_poly2 = poly2.fit_transform(X_clean)
    model_poly2 = LinearRegression()
    model_poly2.fit(X_poly2, y_clean)
    X_sorted_poly2 = poly2.fit_transform(X_sorted)
    y_pred_poly2 = model_poly2.predict(X_sorted_poly2)
    r2_poly2 = r2_score(y_clean, model_poly2.predict(X_poly2))
    
    axes[3].scatter(X_clean, y_clean, alpha=0.3, color='gray', s=10)
    axes[3].plot(X_sorted, y_pred_poly2, 'purple', linewidth=2, label=f'Полином 2-й степени (R²={r2_poly2:.4f})')
    axes[3].set_title(f'Полиномиальная 2-й степени\nR² = {r2_poly2:.4f}')
    axes[3].set_xlabel(x_name)
    axes[3].set_ylabel(y_name)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. ПОЛИНОМИАЛЬНАЯ 3-Й СТЕПЕНИ
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X_clean)
    model_poly3 = LinearRegression()
    model_poly3.fit(X_poly3, y_clean)
    X_sorted_poly3 = poly3.fit_transform(X_sorted)
    y_pred_poly3 = model_poly3.predict(X_sorted_poly3)
    r2_poly3 = r2_score(y_clean, model_poly3.predict(X_poly3))
    
    axes[4].scatter(X_clean, y_clean, alpha=0.3, color='gray', s=10)
    axes[4].plot(X_sorted, y_pred_poly3, 'red', linewidth=2, label=f'Полином 3-й степени (R²={r2_poly3:.4f})')
    axes[4].set_title(f'Полиномиальная 3-й степени\nR² = {r2_poly3:.4f}')
    axes[4].set_xlabel(x_name)
    axes[4].set_ylabel(y_name)
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # 6. ВСЕ МОДЕЛИ ВМЕСТЕ (сравнительный график)
    axes[5].scatter(X_clean, y_clean, alpha=0.3, color='gray', s=10)
    axes[5].plot(X_sorted, y_pred_lin, 'b-', linewidth=1.5, label=f'Линейная (R²={r2_lin:.4f})')
    axes[5].plot(X_sorted, y_pred_exp, 'g-', linewidth=1.5, label=f'Экспоненциальная (R²={r2_exp:.4f})')
    axes[5].plot(X_sorted, y_pred_log, 'orange', linewidth=1.5, label=f'Логарифмическая (R²={r2_log:.4f})')
    axes[5].plot(X_sorted, y_pred_poly2, 'purple', linewidth=1.5, label=f'Полином 2 (R²={r2_poly2:.4f})')
    axes[5].plot(X_sorted, y_pred_poly3, 'red', linewidth=1.5, label=f'Полином 3 (R²={r2_poly3:.4f})')
    axes[5].set_title(f'Сравнение всех моделей\nЛучшая: {max([(r2_lin,"Линейная"),(r2_exp,"Экспоненциальная"),(r2_log,"Логарифмическая"),(r2_poly2,"Полином 2"),(r2_poly3,"Полином 3)"], key=lambda x: x[0])[1]}')
    axes[5].set_xlabel(x_name)
    axes[5].set_ylabel(y_name)
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle(f'{y_name} от {x_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"  График сохранён: {output_filename}")
    
    # Вывод результатов сравнения
    print(f"\n  Результаты для пары {x_name} → {y_name}:")
    print(f"    Линейная: R² = {r2_lin:.4f}")
    print(f"    Экспоненциальная: R² = {r2_exp:.4f}")
    print(f"    Логарифмическая: R² = {r2_log:.4f}")
    print(f"    Полином 2-й степени: R² = {r2_poly2:.4f}")
    print(f"    Полином 3-й степени: R² = {r2_poly3:.4f}")
    
    # Лучшая модель
    best_r2 = max([r2_lin, r2_exp, r2_log, r2_poly2, r2_poly3])
    if best_r2 == r2_lin:
        best_name = "Линейная"
    elif best_r2 == r2_exp:
        best_name = "Экспоненциальная"
    elif best_r2 == r2_log:
        best_name = "Логарифмическая"
    elif best_r2 == r2_poly2:
        best_name = "Полиномиальная 2-й степени"
    else:
        best_name = "Полиномиальная 3-й степени"
    
    print(f"    ЛУЧШАЯ МОДЕЛЬ: {best_name} (R² = {best_r2:.4f})")
    
    return {'best_model': best_name, 'best_r2': best_r2}

def main():
    print("=" * 80)
    print("ПОСТРОЕНИЕ ВСЕХ ТИПОВ РЕГРЕССИИ ДЛЯ КОРРЕЛИРУЮЩИХ ПАР")
    print("=" * 80)
    
    df = load_and_prepare_data()
    
    # Названия показателей
    names = {
        'ВРП': 'ВРП на душу (руб.)',
        'Инвестиции': 'Инвестиции на душу (руб.)',
        'Доходы': 'Доходы населения (руб.)',
        'Продолжительность_жизни': 'Продолжительность жизни (лет)',
        'Уровень_бедности': 'Уровень бедности (%)',
        'Жилплощадь': 'Жилплощадь на человека (кв.м)'
    }
    
    # Пары для анализа (все возможные комбинации факторов и результатов)
    # Факторы: ВРП, Инвестиции, Доходы
    # Результаты: Продолжительность_жизни, Уровень_бедности, Жилплощадь
    
    factors = ['ВРП', 'Инвестиции', 'Доходы']
    targets = ['Продолжительность_жизни', 'Уровень_бедности', 'Жилплощадь']
    
    print("\nПары для анализа:")
    print("-" * 50)
    
    results_summary = []
    
    for target in targets:
        for factor in factors:
            print(f"  {target} ← {factor}")
            X = df[factor].values
            y = df[target].values
            x_name = names[factor]
            y_name = names[target]
            filename = f'results/figures/regression_{target}_from_{factor}.png'
            
            res = plot_all_regression_types(X, y, x_name, y_name, filename)
            results_summary.append({
                'Зависимая': target,
                'Фактор': factor,
                'Лучшая модель': res['best_model'],
                'R²': res['best_r2']
            })
    
    # Сводная таблица результатов
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))
    
    # Сохраняем результаты
    results_df.to_csv('results/tables/models_comparison_summary.csv', index=False)
    print("\n✅ Результаты сохранены в results/tables/models_comparison_summary.csv")

if __name__ == "__main__":
    main()
