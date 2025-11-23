import matplotlib.pyplot as plt

# ==========================================
# 1. INPUT SECTION (Edit this part only)
# ==========================================
experiment_names = ['C_Entropy', 'L_Smoothing', 'Hinge_Loss']
accuracy_scores  = [58.23, 58.36, 57.88]  # percentages (0-100)

chart_title = "Model Accuracy Comparison"

# NEW: Control the plot window (scale) here
y_axis_min = 55  # The bottom of the graph
y_axis_max = 60  # The top of the graph
# ==========================================


# 2. Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(experiment_names, accuracy_scores, color="#EAE325")

# 3. Formatting
plt.title(chart_title, fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Experiments', fontsize=12)

# 4. Apply the custom scale (The "Zoom")
plt.ylim(y_axis_min, y_axis_max)

# 5. Add the data labels
for bar in bars:
    height = bar.get_height()
    # If the bar is taller than the max scale, the text might get cut off,
    # but for typical zooming, this places the label right above the bar.
    plt.text(bar.get_x() + bar.get_width()/2., height + (y_axis_max - y_axis_min)*0.01,
             f'{height}%',
             ha='center', va='bottom', fontweight='bold')

# 6. Display
plt.tight_layout()
plt.show()