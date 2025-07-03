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
Eo = st.number_input("Eo", value=100.0, step=0.1)
Fi_input = st.number_input("Fi (ϕ) стойност", value=0.3, step=0.01)

# Въвеждане на h_i и E_i за всеки пласт
st.markdown("### Въведи стойности за всеки пласт")
h_values = []
E_values = []
cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E{to_subscript(i+1)}", value=1000.0, step=0.1, key=f"E_{i}")
        E_values.append(E)

h_array = np.array(h_values)
E_array = np.array(E_values)

# Изчисляване на H и Esr
H = h_array.sum()
weighted_sum = np.sum(E_array * h_array)
Esr = weighted_sum / H if H != 0 else 0

# Формули и резултати (с latex)
st.latex(r"H = \sum_{i=1}^n h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
st.latex(r"H = " + h_terms)
st.write(f"H = {H:.3f}")

st.latex(r"Esr = \frac{\sum_{i=1}^n (E_i \cdot h_i)}{\sum_{i=1}^n h_i}")
numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum:.3f}}}{{{H:.3f}}} = {Esr:.3f}"
st.latex(formula_with_values)

ratio = H / D if D != 0 else 0
st.latex(r"\frac{H}{D} = \frac{" + f"{H:.3f}" + "}{" + f"{D}" + "} = " + f"{ratio:.3f}")

st.latex(r"\frac{Esr}{E_o} = \frac{" + f"{Esr:.3f}" + "}{" + f"{Eo}" + "} = " + f"{Esr / Eo:.3f}")
Esr_over_Eo = Esr / Eo if Eo != 0 else 0

# Зареждане на данни
df_fi = pd.read_csv("fi.csv", delimiter=';')
df_esr_eo = pd.read_csv("Esr_Eo.csv", delimiter=';')

df_fi.rename(columns={df_fi.columns[2]: 'fi'}, inplace=True)
df_esr_eo.rename(columns={df_esr_eo.columns[2]: 'Esr_Eo'}, inplace=True)

fig = go.Figure()

# Изолинии fi
unique_fi = sorted(df_fi['fi'].unique())
for fi_val in unique_fi:
    df_level = df_fi[df_fi['fi'] == fi_val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'ϕ = {fi_val}',
        line=dict(width=2)
    ))

# Изолинии Esr/Eo
unique_esr_eo = sorted(df_esr_eo['Esr_Eo'].unique())
for val in unique_esr_eo:
    df_level = df_esr_eo[df_esr_eo['Esr_Eo'] == val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'Esr/Eo = {val}',
        line=dict(width=2)
    ))

# --- ЧЕРВЕНА ТОЧКА (вертикална линия от H/D до пресичане с Esr/Eo изолиния)
closest_val = min(unique_esr_eo, key=lambda x: abs(x - Esr_over_Eo))
df_closest = df_esr_eo[df_esr_eo['Esr_Eo'] == closest_val].sort_values(by='H/D')

def get_point_on_curve(df, x_target):
    x_vals = df['H/D'].values
    y_vals = df['y'].values
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x_target <= x_vals[i + 1]:
            x1, y1 = x_vals[i], y_vals[i]
            x2, y2 = x_vals[i + 1], y_vals[i + 1]
            t = (x_target - x1) / (x2 - x1)
            y_interp = y1 + t * (y2 - y1)
            return np.array([x_target, y_interp])
    return None

point_on_esr_eo = get_point_on_curve(df_closest, ratio)

if point_on_esr_eo is not None:
    # Червена точка
    fig.add_trace(go.Scatter(
        x=[point_on_esr_eo[0]],
        y=[point_on_esr_eo[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Червена точка (Esr/Eo)'
    ))
    # Вертикална линия от H/D до Esr/Eo
    fig.add_trace(go.Scatter(
        x=[point_on_esr_eo[0], point_on_esr_eo[0]],
        y=[0, point_on_esr_eo[1]],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Вертикална линия H/D → Esr/Eo'
    ))

    # --- ОРАНЖЕВА ТОЧКА (проекция хоризонтално към fi)
    # Намерение на хоризонталната линия от червената точка до пресичане с fi
    y_target = point_on_esr_eo[1]

    # Функция за намиране на x за дадено y по дадена линия fi
    def interp_x_for_y(df, y_target):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for k in range(len(y_arr) - 1):
            y1, y2 = y_arr[k], y_arr[k + 1]
            if (y1 - y_target) * (y2 - y_target) <= 0:
                x1, x2 = x_arr[k], x_arr[k + 1]
                if y2 == y1:
                    return x1
                t_local = (y_target - y1) / (y2 - y1)
                return x1 + t_local * (x2 - x1)
        return None

    # Намерение на интерполация между две fi линии около Fi_input
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

        x_fi_lower = interp_x_for_y(df_fi_lower, y_target)
        x_fi_upper = interp_x_for_y(df_fi_upper, y_target)

        if x_fi_lower is not None and x_fi_upper is not None:
            t_fi = (Fi_input - fi_lower_val) / (fi_upper_val - fi_lower_val)
            x_fi_interp = x_fi_lower + t_fi * (x_fi_upper - x_fi_lower)

            # Оранжева точка
            fig.add_trace(go.Scatter(
                x=[x_fi_interp],
                y=[y_target],
                mode='markers',
                marker=dict(color='orange', size=10),
                name='Оранжева точка'
            ))

            # Вертикална линия от оранжевата точка до y=1.35
            fig.add_trace(go.Scatter(
                x=[x_fi_interp, x_fi_interp],
                y=[y_target, 1.35],
                mode='lines',
                line=dict(color='orange', dash='dash'),
                name='Вертикална линия до y=1.35'
            ))

# Настройки на графиката
fig.update_layout(
    xaxis_title="H/D",
    yaxis_title="y",
    legend_title="Изолинии",
    yaxis=dict(range=[0, max(1.5, df_fi['y'].max())]),
    width=800,
    height=600,
    margin=dict(l=50, r=50, t=50, b=50)
)

st.plotly_chart(fig, use_container_width=True)
