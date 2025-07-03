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

# Формули и резултати
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
df_fi = pd.read_csv("fi.csv")
df_esr_eo = pd.read_csv("Esr_Eo.csv")

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

# Функция за линейна интерполация на точка (x_target) по H/D за дадена крива
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

# --- ЧЕРВЕНА ТОЧКА и вертикална червена линия от x=H/D до y на червената точка
closest_val = min(unique_esr_eo, key=lambda x: abs(x - Esr_over_Eo))
df_closest = df_esr_eo[df_esr_eo['Esr_Eo'] == closest_val].sort_values(by='H/D')

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
    
    # Вертикална линия от x=H/D от 0 до y на червената точка
    fig.add_trace(go.Scatter(
        x=[ratio, ratio],
        y=[0, point_on_esr_eo[1]],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Вертикална линия H/D → Esr/Eo (червена)'
    ))
    
    # --- ОРАНЖЕВА ТОЧКА и хоризонтална линия от червената точка до пресичане с Fi_input по y
    y_red = point_on_esr_eo[1]
    
    # Функция за намиране на x (H/D) за дадена y и Fi
    def interp_x_for_fi(df, fi_target, y_target):
        df_fi_target = df[df['fi'] == fi_target]
        x_arr = df_fi_target['H/D'].values
        y_arr = df_fi_target['y'].values
        
        for k in range(len(y_arr)-1):
            y1, y2 = y_arr[k], y_arr[k+1]
            if (y1 - y_target)*(y2 - y_target) <= 0:
                x1, x2 = x_arr[k], x_arr[k+1]
                if y2 == y1:
                    return x1
                t = (y_target - y1)/(y2 - y1)
                return x1 + t*(x2 - x1)
        return None

    # Проверка дали Fi_input съществува в df_fi['fi'], ако не - използваме най-близката стойност
    if Fi_input not in df_fi['fi'].values:
        Fi_input = min(df_fi['fi'].unique(), key=lambda x: abs(x - Fi_input))

    x_orange = interp_x_for_fi(df_fi, Fi_input, y_red)
    
    if x_orange is not None:
        # Оранжева точка
        fig.add_trace(go.Scatter(
            x=[x_orange],
            y=[y_red],
            mode='markers',
            marker=dict(color='orange', size=10),
            name='Оранжева точка'
        ))
        
        # Хоризонтална линия от червената точка до оранжевата точка
        fig.add_trace(go.Scatter(
            x=[point_on_esr_eo[0], x_orange],
            y=[y_red, y_red],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name='Хоризонтална линия (червена → оранжева)'
        ))
        
        # Вертикална линия от оранжевата точка до y=1.35
        fig.add_trace(go.Scatter(
            x=[x_orange, x_orange],
            y=[y_red            , 1.35],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name='Вертикална линия оранжева точка → y=1.35'
        ))

# Настройка на графиката
fig.update_layout(
    title="Графика на изолинии и точки",
    xaxis_title="H/D",
    yaxis_title="y",
    legend_title="Легенда",
    width=900,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

