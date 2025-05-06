import streamlit as st
import joblib
import numpy as np
import pandas as pd
model = joblib.load("model.pkl")

player_data = pd.read_csv("first_round_with_fixed_trade_count.csv")

st.title("What-if predictor")

player_names = sorted(player_data["Player"].unique())
selected_player = st.selectbox("Select a Player", player_names)
# Get most recent season stats for selected player
player_row = player_data[player_data["Player"] == selected_player].sort_values("SEASON_YEAR", ascending=False).iloc[0]

# Pre-fill sliders
trade_count = st.slider("Number of Trades", 0, 10, value=int(player_row["trade_count_so_far"]))
years_in_league = st.slider("Years in League", 0, 20, value=int(player_row["Years_In_League"]))
player_age = st.slider("Player Age", 18, 40, value=int(player_row["PLAYER_AGE"]))
pts = st.number_input("Points", value=int(player_row["PTS"]))
ast = st.number_input("Assists", value=int(player_row["AST"]))
reb = st.number_input("Rebounds", value=int(player_row["REB"]))
gp = st.number_input("Games Played", value=int(player_row["GP"]))

actual_bpm = player_row["Custom_BPM"]
st.metric("Actual Custom BPM", f"{actual_bpm:.2f}")

features = np.array([[trade_count, years_in_league, player_age, pts, ast, reb, gp]])
pred_bpm = model.predict(features)[0]

st.metric("Predicted Custom BPM", f"{pred_bpm:.2f}")

similar_players = player_data[
    (player_data["trade_count_so_far"] == 0) &
    (player_data["Years_In_League"].between(years_in_league - 1, years_in_league + 1)) &
    (player_data["PTS"].between(pts - 200, pts + 200)) &
    (player_data["AST"].between(ast - 50, ast + 50)) &
    (player_data["REB"].between(reb - 50, reb + 50)) &
    (player_data["GP"].between(gp - 10, gp + 10))
]

if not similar_players.empty:
    st.subheader("📊 Similar Players with 0 Trades")
    
    avg_bpm = similar_players["Custom_BPM"].mean()
    st.metric("Average BPM (Similar No-Trade Players)", f"{avg_bpm:.2f}")

    st.dataframe(similar_players[["Player", "Custom_BPM", "PTS", "AST", "REB", "GP", "Years_In_League"]].sort_values("Custom_BPM", ascending=False))
else:
    st.warning("No similar players with 0 trades found.")
