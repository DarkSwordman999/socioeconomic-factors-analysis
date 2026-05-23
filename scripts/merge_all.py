import pandas as pd

def merge_all():
    gdp = pd.read_csv('data/processed/gdp.csv')
    invest = pd.read_csv('data/processed/investments.csv')
    income = pd.read_csv('data/processed/income.csv')
    poverty = pd.read_csv('data/processed/poverty.csv')
    life = pd.read_csv('data/processed/life_expectancy.csv')
    housing = pd.read_csv('data/processed/housing.csv')
    
    df = gdp.merge(invest, on=['Регион', 'Год'], how='inner')
    df = df.merge(income, on=['Регион', 'Год'], how='inner')
    df = df.merge(poverty, on=['Регион', 'Год'], how='inner')
    df = df.merge(life, on=['Регион', 'Год'], how='inner')
    df = df.merge(housing, on=['Регион', 'Год'], how='inner')
    
    return df

if __name__ == "__main__":
    data = merge_all()
    data.to_csv('data/processed/final_dataset.csv', index=False)
    print(f"Final dataset shape: {data.shape}")
