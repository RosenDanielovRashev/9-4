import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr и H_n/D")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

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

# Изчисляване на Esr за първите n-1 пласта
sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

# Изчисляване на H_n и H_{n-1}
H_n = h_array.sum()
H_n_1 = sum_h_n_1

# Закръгляне
H_n_1_r = round(H_n_1, 3)
H_n_r = round(H_n, 3)
Esr_r = round(Esr, 3)
ratio = H_n / D if D != 0 else 0
ratio_r = round(ratio, 3)

st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
st.latex(r"H_{n-1} = " + h_terms)
st.write(f"H{to_subscript(n-1)} = {H_n_1_r}")

st.latex(r"H_n = \sum_{i=1}^n h_i")
h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
st.latex(r"H_n = " + h_terms_n)
st.write(f"H{to_subscript(n)} = {H_n_r}")

st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

numerator = " + ".join([f"{round(E_values[i],3)} \cdot {round(h_values[i],3)}" for i in range(n-1)])
denominator = " + ".join([f"{round(h_values[i],3)}" for i in range(n-1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr_r}"
st.latex(formula_with_values)

ratio_display = rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}"
st.latex(ratio_display)

Ed = st.number_input("Ed", value=100.0, step=0.1)
Ed_r = round(Ed, 3)

En = E_values[-1]
En_r = round(En, 3)

st.markdown("### Изчисления с последен пласт")

st.latex(r"E_{" + str(n) + r"} = " + f"{En_r}")

Esr_over_En = Esr / En if En != 0 else 0
Esr_over_En_r = round(Esr_over_En, 3)
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr_r}" + "}{" + f"{En_r}" + "} = " + f"{Esr_over_En_r}")

En_over_Ed = En / Ed if Ed != 0 else 0
En_over_Ed_r = round(En_over_Ed, 3)
st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En_r}" + "}{" + f"{Ed_r}" + "} = " + f"{En_over_Ed_r}")

# Зареждане на данни и построяване на графика
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {round(level,3)}',
            line=dict(width=2)
        ))

if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Esr/Ei = {round(sr_Ei,3)}',
            line=dict(width=2)
        ))

# --- Интерполация на y за Esr/Ei и Hn/D

sr_Ei_values = sorted(df_new['sr_Ei'].unique())
target_sr_Ei = Esr_over_En_r
target_Hn_D = ratio_r

y_at_ratio = None
interp_error = False

if target_sr_Ei < sr_Ei_values[0] or target_sr_Ei > sr_Ei_values[-1]:
    st.error(f"❌ Стойността Esr/Ei = {target_sr_Ei} е извън обхвата на наличните изолинии.")
    interp_error = True
else:
    if target_sr_Ei in sr_Ei_values:
        df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
        y_at_ratio = np.interp(target_Hn_D, df_target['H/D'].values, df_target['y'].values)
    else:
        # Намерете индекси на съседни стойности
        lower_idx = None
        upper_idx = None
        for i in range(len(sr_Ei_values) - 1):
            if sr_Ei_values[i] <= target_sr_Ei <= sr_Ei_values[i + 1]:
                lower_idx = i
                upper_idx = i + 1
                break
        if lower_idx is None or upper_idx is None:
            st.error(f"❌ Не може да се намери интервал за Esr/Ei = {target_sr_Ei}")
            interp_error = True
        else:
            lower_val = sr_Ei_values[lower_idx]
            upper_val = sr_Ei_values[upper_idx]

            df_lower = df_new[df_new['sr_Ei'] == lower_val].sort_values(by='H/D')
            df_upper = df_new[df_new['sr_Ei'] == upper_val].sort_values(by='H/D')

            y_lower = np.interp(target_Hn_D, df_lower['H/D'].values, df_lower['y'].values)
            y_upper = np.interp(target_Hn_D, df_upper['H/D'].values, df_upper['y'].values)

            t = (target_sr_Ei - lower_val) / (upper_val - lower_val)
            y_at_ratio = y_lower + t * (y_upper - y_lower)

if not interp_error and y_at_ratio is not None:
    # Добавяне на вертикална линия на Hn/D
    fig.add_trace(go.Scatter(
        x=[ratio_r, ratio_r],
        y=[0, y_at_ratio],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Вертикална линия на Hn/D'
    ))

    # Добавяне на червена точка на пресечната точка (интерполирана точка)
    fig.add_trace(go.Scatter(
        x=[ratio_r],
        y=[y_at_ratio],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка'
    ))

    # --- Търсене на пресечна точка с изолинии Ei/Ed (от df_original)

    def interp_x_for_y(df, y_target):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for k in range(len(y_arr) - 1):
            y1, y2 = y_arr[k], y_arr[k + 1]
            if (y1 - y_target) * (y2 - y_target) <= 0:  # y_target между y1 и y2
                x1, x2 = x_arr[k], x_arr[k + 1]
                if y2 == y1:
                    return x1
                t_local = (y_target - y1) / (y2 - y1)
                x_interp = x1 + t_local * (x2 - x1)
                return x_interp
        return None

    Ei_Ed_target = En_over_Ed_r
    Ei_Ed_values_sorted = sorted(df_original['Ei/Ed'].unique())
    lower_index_EiEd = None

    if Ei_Ed_target < Ei_Ed_values_sorted[0] or Ei_Ed_target > Ei_Ed_values_sorted[-1]:
        st.error(f"❌ Стойността Ei/Ed = {Ei_Ed_target} е извън обхвата на наличните изолинии.")
    else:
        for i in range(len(Ei_Ed_values_sorted) - 1):
            if Ei_Ed_values_sorted[i] <= Ei_Ed_target <= Ei_Ed_values_sorted[i + 1]:
                lower_index_EiEd = i
                break

        if Ei_Ed_target in Ei_Ed_values_sorted:
            df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
            x_intercept = interp_x_for_y(df_level, y_at_ratio)
        elif lower_index_EiEd is not None:
            low_val = Ei_Ed_values_sorted[lower_index_EiEd]
            high_val = Ei_Ed_values_sorted[lower_index_EiEd + 1]

            df_low = df_original[df_original['Ei/Ed'] == low_val].sort_values(by='H/D')
            df_high = df_original[df_original['Ei/Ed'] == high_val].sort_values(by='H/D')

            x_low = interp_x_for_y(df_low, y_at_ratio)
            x_high = interp_x_for_y(df_high, y_at_ratio)

            if x_low is not None and x_high is not None:
                t_EiEd = (Ei_Ed_target - low_val) / (high_val - low_val)
                x_intercept = x_low + t_EiEd * (x_high - x_low)
            else:
                x_intercept = None
        else:
            x_intercept = None

        if x_intercept is not None:
            # Оранжева точка вместо звезда
            fig.add_trace(go.Scatter(
                x=[x_intercept],
                y=[y_at_ratio],
                mode='markers',
                marker=dict(color='orange', size=12, symbol='circle'),
                name='Пресечна точка с Ei/Ed'
            ))

            # Линия свързваща оранжевата точка и червената точка (интерполирана точка)
            fig.add_trace(go.Scatter(
                x=[x_intercept, ratio_r],
                y=[y_at_ratio, y_at_ratio],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False
            ))

            # Вертикална линия от оранжевата точка нагоре до y=2.5
            fig.add_trace(go.Scatter(
                x=[x_intercept, x_intercept],
                y=[y_at_ratio, 2.5],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False
            ))
        else:
            st.warning("⚠️ Не може да се намери пресечна точка с Ei/Ed изолиниите.")

fig.update_layout(
    xaxis_title="Hn/D",
    yaxis_title="y",
    legend_title="Изолинии",
    template="plotly_white",
    width=900,
    height=600
)
# --- Добавяне на невидим trace за втората ос (за да се покаже мащабът)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[None, None],  # y не влияе
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip',
    xaxis='x2'  # Свързваме с втората ос
))

fig.update_layout(
    title='Графика на изолинии',
    xaxis=dict(
        title='H/D',
        showgrid=True,
        zeroline=False,
    ),
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[fig.layout.xaxis.range[0] if fig.layout.xaxis.range else None, 1],
        showgrid=False,
        zeroline=False,
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=['0', '0.25', '0.5', '0.75', '1'],
        title='σr'
    ),
    yaxis=dict(
        title='y',
    ),
    showlegend=False
)
st.plotly_chart(fig)

# Проверка дали x_intercept е дефинирана и не е None
if ('x_intercept' in locals()) and (x_intercept is not None):
    sigma_r = round(x_intercept / 2, 3)
    st.markdown(f"**σr = {sigma_r}**")
else:
    st.markdown("**σr = -** (Няма изчислена стойност)")
