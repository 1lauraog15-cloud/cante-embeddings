import pandas as pd

c100 = pd.read_csv('data/raw/cante100_catalogo - cante100_catalogo.csv.csv')
tonas = pd.read_csv('data/raw/tonas_catalogo - tonas_catalogo.csv.csv')

print(c100.shape)
print(c100.columns.tolist())
print()
print(tonas.shape)
print(tonas.columns.tolist())

import re
import unicodedata

def norm(name):
    if pd.isna(name):
        return ""
    s = str(name)
    s = re.sub(r'\(.*?\)', '', s)          # quita paréntesis, ej. "(nombre de pila sin confirmar)"
    s = s.replace('"', '')                  # quita comillas
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')  # quita acentos
    s = s.lower().strip()
    s = re.sub(r'\bel\b|\bla\b|\bde\b|\bdel\b', '', s)  # quita articulos sueltas
    s = re.sub(r'[^a-z]', '', s)            # se queda solo con letras
    return s

# Prueba rápida para comprobar que funciona
print(norm('Manuel Soto "Sordera"'))
print(norm('M. Vargas (nombre de pila sin confirmar)'))
print(norm('Tía Anica "La Piriñaca"'))

# --- Deduplicar cante100 por hash de audio ---
c100_dedup = c100.drop_duplicates(subset='hash_audio_1mb', keep='first').copy()
c100_dedup['norm_name'] = c100_dedup['cantaor_normalizado'].apply(norm)

print(f"cante100: {len(c100)} pistas -> {len(c100_dedup)} tras quitar audio duplicado")

# --- Filtrar TONAS por confianza de mapeo ---
tonas['norm_name'] = tonas['cantaor_propuesto'].apply(norm)
tonas_fiable = tonas[tonas['confianza_mapeo'].isin(['alta', 'media'])].copy()

print(f"TONAS: {len(tonas)} pistas -> {len(tonas_fiable)} con confianza media/alta (excluye {len(tonas) - len(tonas_fiable)} de confianza baja)")

# --- Contar grabaciones por cantaor en cada dataset ---
c100_counts = c100_dedup.groupby('norm_name').size().rename('n_c100')
c100_names = c100_dedup.groupby('norm_name')['cantaor_normalizado'].first().rename('nombre_c100')

tonas_counts = tonas_fiable.groupby('norm_name').size().rename('n_tonas')
tonas_names = tonas_fiable.groupby('norm_name')['cantaor_propuesto'].first().rename('nombre_tonas')

print(c100_counts.head())
print()
print(tonas_counts.head())

# --- Combinar ambos datasets sumando por nombre normalizado ---
combined = pd.concat([c100_counts, c100_names, tonas_counts, tonas_names], axis=1)
combined['n_c100'] = combined['n_c100'].fillna(0).astype(int)
combined['n_tonas'] = combined['n_tonas'].fillna(0).astype(int)
combined['total'] = combined['n_c100'] + combined['n_tonas']
combined['nombre'] = combined['nombre_c100'].fillna(combined['nombre_tonas'])
combined = combined.sort_values('total', ascending=False)

print(combined[['nombre', 'n_c100', 'n_tonas', 'total']].to_string())

# --- Fusión manual: Rafael Romero / Rafael Romero "El Gallina" (confirmado en notas de TONAS) ---
rr = combined.loc[['rafaelromero', 'rafaelromerogallina']]
merged_row = pd.DataFrame({
    'n_c100': [rr['n_c100'].sum()],
    'nombre_c100': ['Rafael Romero'],
    'n_tonas': [rr['n_tonas'].sum()],
    'nombre_tonas': ['Rafael Romero "El Gallina"'],
    'total': [rr['total'].sum()],
    'nombre': ['Rafael Romero']
}, index=['rafaelromero_merged'])

combined = combined.drop(['rafaelromero', 'rafaelromerogallina'])
combined = pd.concat([combined, merged_row]).sort_values('total', ascending=False)

print(f"Cantaores únicos (tras corregir el cruce Rafael Romero): {len(combined)}")
print()
print(f"Con >=3 grabaciones: {(combined['total'] >= 3).sum()}")
print(f"Con 2 grabaciones:   {(combined['total'] == 2).sum()}")
print(f"Con 1 grabación:     {(combined['total'] == 1).sum()}")

# --- Guardar resultado ---
combined.to_csv('data/processed/conteo_cantaores_combinado.csv')
print("\nGuardado en data/processed/conteo_cantaores_combinado.csv")