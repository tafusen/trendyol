import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing


def main(verbosity=False):
    st.header("Build a Mixture Regression Model")
    st.markdown("""
    In this we do session, we will be implementing a mixture linear regressor:
    
    1. A general linear regression model
    2. A **building age** level specific model.
    
    Ultimate loss function will be optimizing the parameters of both models at the same time
    """)

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    st.dataframe(X)
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))

    st.dataframe(df)
    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Formulating the mixture Group")

    st.markdown("#### General Model")
    st.latex(r"\hat{y}^{0}_i=\beta_0 + \beta_1 x_i")

    st.markdown("###### Loss Function")


    lam = st.slider("Regularization Multiplier for L2 beta (lamda)", 0.001, 1., value=0.1)


    theta =  st.slider("Error Threshold (theta)", 1., 10., value=5.)
    p = 1
    beta = reg(df['MedInc'].values, df['Price'].values, p=p, verbose=verbosity, theta= theta, lam=lam)

    st.subheader(f"General Model with p={p:.2f} contribution")
    st.latex(fr"Price = {beta[1]:.4f} \times MedInc + {beta[0]:.4f}")
    st.latex(fr"Error = {err}")

def reg(x, y, p=1, verbose=False,theta = 0.05, lam=0.1):
    beta = np.random.random(2)  

    if verbose:
        st.write(beta)
        st.write(x)
    alpha = 0.0001
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _x, _y in zip( x, y):
            y_pred = p * (beta[0] + beta[1] * _x)
            thetat = theta
            if(abs(_y-y_pred) > theta):
                if(_y-y_pred < 0):
                    thetat = -theta

                g_b0 = -2  * (thetat) + 2 * lam * beta[0]
                g_b1 = -2  * ((thetat) * _x) *lam * beta[1]
                err += (thetat) ** 2

                # st.write(f"Gradient of beta0: {g_b0}")

               

            else:
                g_b0 = -2 * (_y - y_pred) + 2 * lam * beta[0]
                g_b1 = -2 * ((_y - y_pred) * _x) + 2 * lam * beta[1]
                err += (_y - y_pred) ** 2
                # st.write(f"Gradient of beta0: {g_b0}")

            beta_prev = np.copy(beta)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1
        

        print(f"{it} - Beta: {beta}, Error: {err}")
        my_bar.progress(it / n_max_iter)


    return beta


if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))