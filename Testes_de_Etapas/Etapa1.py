""" Teste completo da Etapa 1: loader -> splitter -> preprocessor """

from src.data.loader import carregar_e_preparar
from src.data.splitter import preparar_split_2025, preparar_teste_2026
from src.data.preprocessor import preparar_features

print("=" * 60)
print("1. Carregamento (amostra 5k linhas)")
print("=" * 60)

df2025 = carregar_e_preparar(ano= 2025, linhas= 5000)
df2026 = carregar_e_preparar(ano= 2026, linhas= 5000)

print(f"2025: {df2025.shape} | target = {df2025['target'].value_counts}")
print(f"2026: {df2026.shape} | target = {df2026['target'].value_counts}")

print("\n" + "=" * 60)
print("2. DIVISÃO (TREINO - VALIDAÇÃO - TESTE) + CONSOLIDAÇÃO BO")
print("\n" + "=" * 60)

train, val, test2025 = preparar_split_2025(df2025)
print(f"Treino: {train.shape} | Val: {val.shape} | Teste: {test2025.shape}")

print("=" * 60)
print("3. Pré-processamento (fit no treino, transform no val)")
print("=" * 60)

X_train, y_train, preprocessor = preparar_features(train, fit=True)
X_val, y_val, _ = preparar_features(val, preprocessor=preprocessor, fit=False)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
print(f"Features: {len(preprocessor.get_feature_names_out())}")

print("=" * 60)
print("TESTE FINAL - 2026 CONSOLIDADO")
print("=" * 60)

teste_2026 = preparar_teste_2026(df2026)
print(f"Teste 2026: {teste_2026.shape}")
print(f"Target:\n{teste_2026['target'].value_counts()}")

print("Etapa 1 Funcionando corretamente")