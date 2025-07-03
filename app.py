import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr и H")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)
Eo = round(st.number_input("Eo", value=100.0, step=0.1), 3)
Fi_input = round(st.number_input("Fi (ϕ) стойност", value=0.3, step=0.01), 3)

# Въвеждане на h_i и E_i за всеки пласт
st.markdown("### Въведи стойности за всеки пласт")
h_values = []
E_values = []
cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = round(st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}"), 3)
        h_values.append(h)
    with cols[1]:
        E = round(st.number_input(f"E{to_subscript(i+1)}", value=1000.0, step=0.1, key=f"E_{i}"), 3)
        E_values.append(E)

h_array = np.array(h_values)
E_array = np.array(E_values)

# Изчисляване на H и Esr за всички n слоя
H = round(h_array.sum(), 3)
weighted_sum = round(np.sum(E_array * h_array), 3)
Esr = round(weighted_sum / H if H != 0 else 0, 3)

st.latex(r"H = \sum_{i=1}^n h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
st.latex(r"H = " + h_terms)
st.write(f"H = {H}")

st.latex(r"Esr = \frac{\sum_{i=1}^n (E_i \cdot h_i)}{\sum_{i=1}^n h_i}")
numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum}}}{{{H}}} = {Esr}"
st.latex(formula_with_values)

ratio = round(H / D if D != 0 else 0, 3)
st.latex(r"\frac{H}{D} = \frac{" + f"{H}" + "}{" + f"{round(D,3)}" + "} = " + f"{ratio}")

En = round(E_values[-1], 3)

st.latex(r"E_{" + str(n) + r"} = " + f"{En}")

Esr_over_En = round(Esr / En if En != 0 else 0, 3)
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr}" + "}{" + f"{En}" + "} = " + f"{Esr_over_En}")

# Зареждане на данни от новите CSV файлове
df_fi = pd.read_csv("fi.csv")  # CSV с колони: y, H/D, fi
df_esr_eo = pd.read_csv("Esr_Eo.csv")  # CSV с колони: y, H/D, Esr/Eo

# Преименуваме колоните за удобство
df_fi.rename(columns={df_fi.columns[2]: 'fi'}, inplace=True)
df_esr_eo.rename(columns={df_esr_eo.columns[2]: 'Esr_Eo'}, inplace=True)

fig = go.Figure()

# Добавяне на изолиниите fi
unique_fi = sorted(df_fi['fi'].unique())
for fi_val in unique_fi:
    df_level = df_fi[df_fi['fi'] == fi_val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'ϕ = {round(fi_val,3)}',
        line=dict(width=2)
    ))

# Добавяне на изолиниите Esr/Eo
unique_esr_eo = sorted(df_esr_eo['Esr_Eo'].unique())
for val in unique_esr_eo:
    df_level = df_esr_eo[df_esr_eo['Esr_Eo'] == val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'Esr/Eo = {round(val,3)}',
        line=dict(width=2)
    ))

# --- Намиране на червената точка (интерполация)

target_sr_Eo = Esr_over_En  # Esr/En
target_H_D = ratio          # H/D

esr_eo_values_sorted = sorted(df_esr_eo['Esr_Eo'].unique())
lower_index = None

for i in range(len(esr_eo_values_sorted) - 1):
    if esr_eo_values_sorted[i] <= target_sr_Eo <= esr_eo_values_sorted[i + 1]:
        lower_index = i
        break

if lower_index is not None:
    lower_val = esr_eo_values_sorted[lower_index]
    upper_val = esr_eo_values_sorted[lower_index + 1]

    df_lower = df_esr_eo[df_esr_eo['Esr_Eo'] == lower_val].sort_values(by='H/D')
    df_upper = df_esr_eo[df_esr_eo['Esr_Eo'] == upper_val].sort_values(by='H/D')

    def interp_xy(df, x0):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for j in range(len(x_arr) - 1):
            if x_arr[j] <= x0 <= x_arr[j + 1]:
                p1 = np.array([x_arr[j], y_arr[j]])
                p2 = np.array([x_arr[j + 1], y_arr[j + 1]])
                t = (x0 - x_arr[j]) / (x_arr[j + 1] - x_arr[j])
                return p1 + t * (p2 - p1)
        if x0 < x_arr[0]:
            return np.array([x_arr[0], y_arr[0]])
        else:
            return np.array([x_arr[-1], y_arr[-1]])

    point_lower = interp_xy(df_lower, target_H_D)
    point_upper = interp_xy(df_upper, target_H_D)

    vec = point_upper - point_lower
    t = (target_sr_Eo - lower_val) / (upper_val - lower_val)
    interp_point = point_lower + t * vec

    interp_point[0] = ratio

    fig.add_trace(go.Scatter(
        x=[interp_point[0]],
        y=[interp_point[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка'
    ))

    fig.add_trace(go.Scatter(
        x=[ratio, ratio],
        y=[0, interp_point[1]],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Вертикална линия до Esr/Eo'
    ))

    def interp_x_for_y(df, y_target, x_col='H/D', y_col='y'):
        x_arr = df[x_col].values
        y_arr = df[y_col].values
        for k in range(len(y_arr) - 1):
            y1, y2 = y_arr[k], y_arr[k + 1]
            if (y1 - y_target) * (y2 - y_target) <= 0:
                x1, x2 = x_arr[k], x_arr[k + 1]
                if y2 == y1:
                    return x1
                t_local = (y_target - y1) / (y2 - y1)
                x_interp = x1 + t_local * (x2 - x1)
                return x_interp
        return None

    fi_values_sorted = sorted(df_fi['fi'].unique())
    lower_index_fi = None
    for i in range(len(fi_values_sorted) - 1):
        if fi_values_sorted[i] <= Fi_input <= fi_values_sorted[i + 1]:
            lower_index_fi = i
            break

    if lower_index_fi is not None:
        fi_lower_val = fi_values_sorted[lower_index_fi]
        fi_upper_val = fi_values_sorted[lower_index_fi + 1]

        df_fi_lower = df_fi[df_fi['fi'] == fi_lower_val].sort_values(by='H/D')
        df_fi_upper = df_fi[df_fi['fi'] == fi_upper_val].sort_values(by='H/D')

        x_fi_lower = interp_x_for_y(df_fi_lower, interp_point[1])
        x_fi_upper = interp_x_for_y(df_fi_upper, interp_point[1])

        if x_fi_lower is not None and x_fi_upper is not None:
            t_fi = (Fi_input - fi_lower_val) / (fi_upper_val - fi_lower_val)
            x_fi_interp = x_fi_lower + t_fi * (x_fi_upper - x_fi_lower)

            orange_x = x_fi_interp
            orange_y = interp_point[1]

            fig.add_trace(go.Scatter(
                x=[orange_x],
                y=[orange_y],
                mode='markers',
                marker=dict(color='orange', size=10),
                name='Оранжева точка'
            ))

            fig.add_trace(go.Scatter(
                x=[orange_x, orange_x],
                y=[orange_y, 1.35],
                mode='lines',
                line=dict(color='orange', dash='dash'),
                name='Вертикална линия от оранжевата точка до y=1.35'
            ))

fig.update_layout(
    xaxis_title="H/D",
    yaxis_title="y",
    legend_title="Изолинии",
    yaxis=dict(range=[0, max(1.5, df_fi['y'].max(), df_esr_eo['y'].max())]),
    xaxis=dict(range=[0, max(df_fi['H/D'].max(), df_esr_eo['H/D'].max())])
)

st.plotly_chart(fig, use_container_width=True)

