import pandas as pd

def load_life_expectancy():
    df = pd.read_excel('data/raw/Раздел 1 - Население.xlsx', sheet_name='1.16.1.', skiprows=5, header=None)
    years = df.iloc[1, 1:].dropna().values
    df = df.iloc[2:, :].copy()
    df.columns = ['Регион'] + list(years)
    df = df.dropna(subset=['Регион'])
    df_long = df.melt(id_vars=['Регион'], var_name='Год', value_name='Продолжительность_жизни')
    df_long = df_long.dropna(subset=['Продолжительность_жизни'])
    df_long['Год'] = df_long['Год'].astype(int)
    return df_long

if __name__ == "__main__":
    data = load_life_expectancy()
    print(f"Life expectancy shape: {data.shape}")
