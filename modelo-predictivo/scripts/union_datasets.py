import pandas as pd

df_sys  = pd.read_csv("modelo-predictivo/dataset/SUM_FISI_SISTEMAS_10_21.csv")
df_soft = pd.read_csv("modelo-predictivo/dataset/SUM_FISI_SOFTWARE_10_21.csv")

df_merged = pd.concat([df_sys, df_soft], ignore_index=True, sort=False)

df_merged.to_csv("SUM_FISI_10_21_merged.csv", index=False)

print(df_sys.shape, df_soft.shape, df_merged.shape)
