import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
#  TASK 02 — Data Cleaning & EDA
#  Dataset : Indian Road Accidents (2018–2023)
#  Source  : Ministry of Road Transport & Highways (MoRTH), India
# ════════════════════════════════════════════════════════════════════

# ── 1. CREATE DATASET (Real MoRTH data) ─────────────────────────────
data_yearly = {
    'Year':      [2018,    2019,    2020,    2021,    2022,    2023],
    'Accidents': [467044,  449002,  366138,  412432,  461312,  480583],
    'Fatalities':[153972,  151113,  131714,  153972,  168491,  172890],
    'Injured':   [469418,  451361,  348279,  384448,  443366,  462825],
}

data_states = {
    'State': [
        'Uttar Pradesh', 'Tamil Nadu', 'Madhya Pradesh', 'Maharashtra',
        'Karnataka', 'Rajasthan', 'Andhra Pradesh', 'Gujarat',
        'Telangana', 'Kerala'
    ],
    'Accidents_2022': [21907, 63681, 54432, 33327, 40664, 22780, 20190, 22692, 14777, 35542],
    'Fatalities_2022':[22595, 17884, 14164, 13548, 11587, 10498,  8146,  7551,  5239,  4314],
    'Injured_2022':   [23390, 59684, 51397, 33914, 46761, 25201, 19741, 27249, 16694, 36229],
}

data_causes = {
    'Cause':      ['Over Speeding', 'Wrong Side Driving', 'Jumping Red Light',
                   'Use of Mobile Phone', 'Drunken Driving', 'Other Causes'],
    'Percentage': [71.2, 5.4, 1.9, 1.6, 1.4, 18.5],
}

data_age = {
    'Age_Group':    ['<18', '18–25', '26–35', '36–45', '46–55', '56–60', '>60'],
    'Deaths_2022':  [ 6423,  34971,  47326,  32184,  22543,   9478,  15566],
}

data_vehicle = {
    'Vehicle_Type': ['Two Wheelers', 'Cars/Jeeps/Taxis', 'Trucks/Lorries',
                     'Buses', 'Pedestrians', 'Others'],
    'Share_Pct':    [44.5, 20.3, 14.2, 4.1, 11.8, 5.1],
}

# ── 2. BUILD DATAFRAMES ──────────────────────────────────────────────
df_yearly  = pd.DataFrame(data_yearly)
df_states  = pd.DataFrame(data_states)
df_causes  = pd.DataFrame(data_causes)
df_age     = pd.DataFrame(data_age)
df_vehicle = pd.DataFrame(data_vehicle)

# ── 3. DATA CLEANING ─────────────────────────────────────────────────
print("=" * 60)
print("  TASK 02 — Indian Road Accidents EDA")
print("  Source: MoRTH Annual Reports 2018–2023")
print("=" * 60)

print("\n[1] Yearly Dataset Shape:", df_yearly.shape)
print(df_yearly.to_string(index=False))

print("\n[2] Missing Values Check:")
print(df_yearly.isnull().sum())
print("→ No missing values found.\n")

# Derived columns
df_yearly['Fatality_Rate']   = (df_yearly['Fatalities'] / df_yearly['Accidents'] * 100).round(2)
df_yearly['Injury_Rate']     = (df_yearly['Injured']    / df_yearly['Accidents'] * 100).round(2)
df_yearly['YoY_Accident_Chg']= df_yearly['Accidents'].pct_change().mul(100).round(2)

print("[3] Cleaned & Enriched Yearly Data:")
print(df_yearly[['Year','Accidents','Fatalities','Injured',
                  'Fatality_Rate','Injury_Rate','YoY_Accident_Chg']].to_string(index=False))

# ── 4. EDA SUMMARY ───────────────────────────────────────────────────
print("\n[4] Key Statistics (2022):")
print(f"  Total Accidents  : {461312:,}")
print(f"  Total Fatalities : {168491:,}")
print(f"  Total Injured    : {443366:,}")
print(f"  Fatality Rate    : {168491/461312*100:.2f}% (deaths per accident)")
print(f"  Top Cause        : Over Speeding (71.2% of fatalities)")
print(f"  Most Affected Age: 26–35 years (47,326 deaths)")
print(f"  Highest Risk State: Tamil Nadu (63,681 accidents)")

# ── 5. VISUALIZATIONS ────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 14), facecolor='#0d0d1a')
fig.suptitle("Indian Road Accidents — Exploratory Data Analysis (2018–2023)\n"
             "Source: Ministry of Road Transport & Highways (MoRTH), India",
             fontsize=15, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

YELLOW = '#F5C518'
BLUE   = '#3A86FF'
RED    = '#FF4757'
GREEN  = '#2ed573'
PINK   = '#FF6B9D'
PURPLE = '#a29bfe'

# ── Chart 1: Yearly trend (accidents, fatalities, injured) ───────────
ax1 = fig.add_subplot(gs[0, :2])
x = df_yearly['Year']
ax1.plot(x, df_yearly['Accidents']/1000,   marker='o', color=YELLOW, linewidth=2.5, label='Accidents')
ax1.plot(x, df_yearly['Fatalities']/1000,  marker='s', color=RED,    linewidth=2.5, label='Fatalities')
ax1.plot(x, df_yearly['Injured']/1000,     marker='^', color=BLUE,   linewidth=2.5, label='Injured')
ax1.fill_between(x, df_yearly['Accidents']/1000, alpha=0.08, color=YELLOW)
ax1.set_title('Yearly Trend: Accidents, Fatalities & Injuries (in Thousands)', color='white', fontsize=11)
ax1.set_xlabel('Year', color='#aaaacc')
ax1.set_ylabel('Count (Thousands)', color='#aaaacc')
ax1.tick_params(colors='#cccccc')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax1.grid(color='#2a2a4a', linestyle='--', linewidth=0.6)
for spine in ax1.spines.values(): spine.set_color('#333355')

# ── Chart 2: Fatality Rate trend ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(df_yearly['Year'], df_yearly['Fatality_Rate'],
        color=[RED if r == df_yearly['Fatality_Rate'].max() else PURPLE
               for r in df_yearly['Fatality_Rate']], edgecolor='#0d0d1a', width=0.6)
ax2.set_title('Fatality Rate\n(Deaths per 100 Accidents)', color='white', fontsize=10)
ax2.set_xlabel('Year', color='#aaaacc')
ax2.set_ylabel('%', color='#aaaacc')
ax2.tick_params(colors='#cccccc', labelsize=8)
ax2.grid(color='#2a2a4a', linestyle='--', linewidth=0.6, axis='y')
for spine in ax2.spines.values(): spine.set_color('#333355')
for bar, val in zip(ax2.patches, df_yearly['Fatality_Rate']):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'{val}%', ha='center', color='white', fontsize=8)

# ── Chart 3: Top 10 States — Accidents 2022 ──────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
states_sorted = df_states.sort_values('Accidents_2022', ascending=True)
colors_bar = [RED if s == 'Tamil Nadu' else BLUE for s in states_sorted['State']]
bars = ax3.barh(states_sorted['State'], states_sorted['Accidents_2022'],
                color=colors_bar, edgecolor='#0d0d1a', height=0.6)
ax3.set_title('Top 10 States by Road Accidents (2022)', color='white', fontsize=11)
ax3.set_xlabel('Number of Accidents', color='#aaaacc')
ax3.tick_params(colors='#cccccc', labelsize=9)
ax3.grid(color='#2a2a4a', linestyle='--', linewidth=0.6, axis='x')
for spine in ax3.spines.values(): spine.set_color('#333355')
for bar, val in zip(bars, states_sorted['Accidents_2022']):
    ax3.text(bar.get_width()+300, bar.get_y()+bar.get_height()/2,
             f'{val:,}', va='center', color='white', fontsize=8)

# ── Chart 4: Causes of Accidents ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
wedge_colors = [RED, BLUE, YELLOW, GREEN, PINK, PURPLE]
wedges, texts, autotexts = ax4.pie(
    df_causes['Percentage'], labels=df_causes['Cause'],
    colors=wedge_colors, autopct='%1.1f%%',
    startangle=140, pctdistance=0.78,
    wedgeprops=dict(edgecolor='#0d0d1a', linewidth=1.5))
for t in texts: t.set_color('white'); t.set_fontsize(7)
for at in autotexts: at.set_color('white'); at.set_fontsize(7.5)
ax4.set_title('Causes of Road Accidents\n(2022)', color='white', fontsize=10)

# ── Chart 5: Deaths by Age Group ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
bar_colors = [RED if ag == '26–35' else BLUE for ag in df_age['Age_Group']]
bars5 = ax5.bar(df_age['Age_Group'], df_age['Deaths_2022'],
                color=bar_colors, edgecolor='#0d0d1a', width=0.6)
ax5.set_title('Road Accident Deaths by Age Group (2022)', color='white', fontsize=11)
ax5.set_xlabel('Age Group (Years)', color='#aaaacc')
ax5.set_ylabel('Number of Deaths', color='#aaaacc')
ax5.tick_params(colors='#cccccc', labelsize=9)
ax5.grid(color='#2a2a4a', linestyle='--', linewidth=0.6, axis='y')
for spine in ax5.spines.values(): spine.set_color('#333355')
for bar, val in zip(bars5, df_age['Deaths_2022']):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
             f'{val:,}', ha='center', color='white', fontsize=8)

# ── Chart 6: Vehicle type involvement ────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.barh(df_vehicle['Vehicle_Type'], df_vehicle['Share_Pct'],
         color=[RED, BLUE, YELLOW, GREEN, PINK, PURPLE],
         edgecolor='#0d0d1a', height=0.6)
ax6.set_title('Accident Share by\nVehicle Type (2022)', color='white', fontsize=10)
ax6.set_xlabel('Share (%)', color='#aaaacc')
ax6.tick_params(colors='#cccccc', labelsize=8)
ax6.grid(color='#2a2a4a', linestyle='--', linewidth=0.6, axis='x')
for spine in ax6.spines.values(): spine.set_color('#333355')
for i, val in enumerate(df_vehicle['Share_Pct']):
    ax6.text(val+0.3, i, f'{val}%', va='center', color='white', fontsize=8)

plt.savefig('india_road_accidents_eda.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("\nVisualization saved as 'india_road_accidents_eda.png'")
