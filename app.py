import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Нови изолинии с Fi, Esr/Eo и точки по нова логика")

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)
Eo = st.number_input("Eo (нов параметър)", value=100.0, step=0.1)
Fi_input = st.number_input("Fi (ϕ) стойност", value=0.3, step=0.01)

# Въвеждане на h_i и E_i за всеки пласт
st.markdown("### Въведи стойности за всеки пласт")
h_values, E_values = [], []
cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h_values.append(st.number_input(f"h{i+1}", value=4.0, step=0.1, key=f"h_{i}"))
    with cols[1]:
        E_values.append(st.number_input(f"E{i+1}", value=1000.0, step=0.1, key=f"E_{i}"))

h_array = np.array(h_values)
E_array = np.array(E_values)

H = h_array.sum()
weighted_sum = np.sum(E_array * h_array)
Esr = weighted_sum / H if H != 0 else 0
H_over_D = H / D if D != 0 else 0
Esr_over_Eo = Esr / Eo if Eo != 0 else 0

# Зареждане на данни
fi_df = pd.read_csv("fi.csv", sep=None, engine='python')
esr_df = pd.read_csv("Esr_Eo.csv", sep=None, engine='python')
fi_df.rename(columns={fi_df.columns[2]: 'fi'}, inplace=True)
esr_df.rename(columns={esr_df.columns[2]: 'Esr_Eo'}, inplace=True)

fig = go.Figure()

# Изолинии fi
for fi_val in sorted(fi_df['fi'].unique()):
    df = fi_df[fi_df['fi'] == fi_val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df['H/D'], y=df['y'], mode='lines', name=f'ϕ = {fi_val}', line=dict(width=2)))

# Изолинии Esr/Eo
for val in sorted(esr_df['Esr_Eo'].unique()):
    df = esr_df[esr_df['Esr_Eo'] == val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df['H/D'], y=df['y'], mode='lines', name=f'Esr/Eo = {val}', line=dict(width=2)))

# ЧЕРВЕНА ТОЧКА: вертикала от H/D до изолинията с най-близко Esr/Eo
def interpolate_y(df, x_target):
    x = df['H/D'].values
    y = df['y'].values
    for i in range(len(x)-1):
        if x[i] <= x_target <= x[i+1]:
            t = (x_target - x[i]) / (x[i+1] - x[i])
            return y[i] + t * (y[i+1] - y[i])
    return None

closest_val = min(esr_df['Esr_Eo'].unique(), key=lambda x: abs(x - Esr_over_Eo))
esr_sub = esr_df[esr_df['Esr_Eo'] == closest_val].sort_values(by='H/D')
y_red = interpolate_y(esr_sub, H_over_D)

if y_red is not None:
    fig.add_trace(go.Scatter(x=[H_over_D], y=[y_red], mode='markers',
        marker=dict(color='red', size=10), name='Червена точка'))
    fig.add_trace(go.Scatter(x=[H_over_D, H_over_D], y=[0, y_red], mode='lines',
        line=dict(color='red', dash='dash'), name='Вертикала от H/D'))

    # ОРАНЖЕВА ТОЧКА: хоризонтала към зададен Fi, намираме пресичане
    fi_vals = sorted(fi_df['fi'].unique())
    lower_fi_idx = next((i for i in range(len(fi_vals)-1) if fi_vals[i] <= Fi_input <= fi_vals[i+1]), None)

    if lower_fi_idx is not None:
        fi1, fi2 = fi_vals[lower_fi_idx], fi_vals[lower_fi_idx+1]
        df1 = fi_df[fi_df['fi'] == fi1].sort_values(by='y')
        df2 = fi_df[fi_df['fi'] == fi2].sort_values(by='y')

        def interpolate_x(df, y_target):
            y = df['y'].values
            x = df['H/D'].values
            for i in range(len(y)-1):
                if y[i] <= y_target <= y[i+1] or y[i] >= y_target >= y[i+1]:
                    t = (y_target - y[i]) / (y[i+1] - y[i])
                    return x[i] + t * (x[i+1] - x[i])
            return None

        x1 = interpolate_x(df1, y_red)
        x2 = interpolate_x(df2, y_red)

        if x1 is not None and x2 is not None:
            t = (Fi_input - fi1) / (fi2 - fi1)
            x_orange = x1 + t * (x2 - x1)

            fig.add_trace(go.Scatter(x=[x_orange], y=[y_red], mode='markers',
                marker=dict(color='orange', size=10), name='Оранжева точка'))
            fig.add_trace(go.Scatter(x=[x_orange, x_orange], y=[y_red, 1.35], mode='lines',
                line=dict(color='orange', dash='dash'), name='Вертикала до y=1.35'))

fig.update_layout(
    xaxis_title="H/D",
    yaxis_title="y",
    yaxis=dict(range=[0, max(1.5, fi_df['y'].max(), esr_df['y'].max())]),
    xaxis=dict(range=[0, max(fi_df['H/D'].max(), esr_df['H/D'].max())]),
    legend_title="Изолинии"
)

st.plotly_chart(fig, use_container_width=True)
