import pandas as pd
df = pd.read_csv("outputs/risk_scoring/powerbi_master_export.csv")
print(df["risk_tier"].value_counts())
print(f"\nCritical+High exposure: ₹{df[df['risk_tier'].isin(['Critical','High'])]['invoice_amount'].sum():,.0f}")
print(f"Top anomaly type: {df[df['anomaly_flag']==1]['anomaly_type'].value_counts().index[0]}")