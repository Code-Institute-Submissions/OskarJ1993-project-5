import streamlit as st
from hypothesis import *
from summary import *
from model_performance import *
from price_study import *
from predict_sales_price import *


class MultiPage:
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(page_title=self.app_name, page_icon="🔑")

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio(
            "Menu", self.pages, format_func=lambda page: page["title"]
        )
        page["function"]()

    def summary(self):
        summary_body()

    def price_study(self):
        price_study()

    def predict_sales_price(self):
        predict_sales_price()

    def hypothesis(self):
        hypothesis_body()

    def model_performance(self):
        model_performance()


app = MultiPage("Sales Price Prediction")
app.add_page("Summary", app.summary)
app.add_page("Price Study", app.price_study)
app.add_page("Predict Sales Price", app.predict_sales_price)
app.add_page("Hypothesis", app.hypothesis)
app.add_page("Model Performance", app.model_performance)

app.run()