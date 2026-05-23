import pandas as pd

def load_housing():
    df = pd.read_excel('data/raw/Раздел 3 - Уровень жизни населения.xlsx', sheet_name='3.25.', skiprows=6, header=None)
    years = df.iloc[0, :].dropna().values
    df = df.iloc[1:, :].copy()
    df.columns = ['Регион'] + list(years)
    df = df.dropna(subset=['Регион'])
    df_long = df.melt(id_vars=['Регион'], var_name='Год', value_name='Жилплощадь')
    df_long = df_long.dropna(subset=['Жилплощадь'])
    df_long['Год'] = df_long['Год'].astype(int)
    return df_long

if __name__ == "__main__":
    data = load_housing()
    print(f"Housing shape: {data.shape}")
