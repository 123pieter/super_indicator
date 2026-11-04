"""Application entrypoint: shared settings and navigation only."""
import streamlit as st
import config
from timeframes import TIMEFRAMES
from providers import PROVIDERS
from ui_pages import scanner, coin_detail, data_management, indicator_backtest

st.set_page_config(page_title="Super Indicator", page_icon=":material/monitoring:", layout="wide")
scanner_page = st.Page(scanner, title="Scanner", icon=":material/table_chart:", default=True)
detail_page = st.Page(coin_detail, title="Coin detail", icon=":material/candlestick_chart:")
data_page = st.Page(data_management, title="Data management", icon=":material/database:")
backtest_page = st.Page(indicator_backtest, title="Indicator backtest", icon=":material/science:")
st.session_state.detail_page = detail_page
page = st.navigation([scanner_page, detail_page, backtest_page, data_page])
with st.sidebar:
    st.title("Super Indicator")
    st.caption("Crypto research workspace")
    st.selectbox("Data provider", list(PROVIDERS), index=PROVIDERS.index(config.ACTIVE_PROVIDER), key="provider")
    st.selectbox("Timeframe", list(TIMEFRAMES), index=list(TIMEFRAMES).index(config.DEFAULT_TIMEFRAME), key="timeframe")
    st.slider("Buy threshold", .1, 1., float(config.BUY_THRESHOLD), .1, key="threshold")
    st.caption("Indicator periods and holding limits count bars on the selected timeframe.")
    st.divider()
    st.caption(f"Storage: {config.LOAD_FROM.upper()}")
page.run()
