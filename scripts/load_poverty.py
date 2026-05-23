import pandas as pd

def load_poverty():
    df = pd.read_excel('data/raw/Раздел 3 - Уровень жизни населения.xlsx', sheet_name='3.12.', skiprows=6, header=None)
    years = df.iloc[0, :].dropna().values
    df = df.iloc[1:, :].copy()
    df.columns = ['Регион'] + list(years)
    df = df.dropna(subset=['Регион'])
    df_long = df.melt(id_vars=['Регион'], var_name='Год', value_name='Уровень_бедности')
    df_long = df_long.dropna(subset=['Уровень_бедности'])
    df_long['Год'] = df_long['Год'].astype(int)
    return df_long

if __name__ == "__main__":
    data = load_poverty()
    print(f"Poverty shape: {data.shape}")
