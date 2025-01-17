import plotly.express as px

months=["January", "February", "March", "April", "May"]
temperatures=[30,35,40,50,55]

fig = px.line(months, temperatures, color_discrete_sequence=['blue'])
fig.update_traces(line=dict(width=5))
fig.update_layout(
    title="Monthly Average Temperatures",
    xaxis_title="Month",
    yaxis_title="Temperature (F)"
)
fig.show()