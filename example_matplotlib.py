import matplotlib.pyplot as plt

months=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
average_rainfall=[78, 60, 92, 55, 110, 95, 75, 65, 85, 70, 80, 90]

plt.bar(months, average_rainfall)

plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.title("Average Monthly Rainfalls")

plt.show()