import pickle
import pandas as pd
import json

# Load segments
with open('data/processed/regime_segments_ma_vix_3regime.pkl', 'rb') as f:
    segs = pickle.load(f)

# Load raw data to check Ukraine period
df = pd.read_csv('data/processed/sp500_vix_merged_clean.csv', parse_dates=['Date'])

# Filter R2 segments
r2_segs = [s for s in segs if s['regime_label'] == 2]

print("=" * 60)
print(f"REGIME 2 (CRISIS) SEGMENTS: {len(r2_segs)} total")
print("=" * 60)

for i, seg in enumerate(r2_segs):
    print(f"\nSegment {i+1}:")
    print(f"  Dates: {seg['start_date']} to {seg['end_date']}")
    print(f"  Length: {seg['length']} days")

# Check Ukraine-Russia war period (Feb 2022)
print("\n" + "=" * 60)
print("UKRAINE-RUSSIA WAR PERIOD CHECK (Feb-Apr 2022)")
print("=" * 60)

ukraine_period = df[(df['Date'] >= '2022-02-01') & (df['Date'] <= '2022-04-30')]

if len(ukraine_period) > 0:
    print(f"Max VIX: {ukraine_period['VIX_Close'].max():.2f}")
    print(f"Days VIX > 40: {(ukraine_period['VIX_Close'] > 40).sum()}")
    print(f"Days VIX > 30: {(ukraine_period['VIX_Close'] > 30).sum()}")
    print(f"Days VIX > 25: {(ukraine_period['VIX_Close'] > 25).sum()}")
    
    # Show actual invasion date
    invasion_date = df[df['Date'] == '2022-02-24']
    if len(invasion_date) > 0:
        print(f"\nFeb 24, 2022 (Invasion day):")
        print(f"  VIX: {invasion_date['VIX_Close'].values[0]:.2f}")
        print(f"  S&P Return: {invasion_date['SP500_Return'].values[0]:.4f}")
else:
    print("⚠️  No data found for Ukraine period")

# Check other potential crises
print("\n" + "=" * 60)
print("OTHER POTENTIAL CRISIS PERIODS")
print("=" * 60)

crisis_periods = {
    '1997 Asian Crisis': ('1997-10-01', '1997-11-30'),
    '1998 LTCM Crisis': ('1998-08-01', '1998-10-31'),
    '2001 Dot-com + 9/11': ('2001-09-01', '2001-10-31'),
    '2011 Euro Crisis': ('2011-08-01', '2011-09-30'),
    '2015 China Crash': ('2015-08-01', '2015-09-30'),
    '2018 Dec Correction': ('2018-12-01', '2018-12-31'),
}

for name, (start, end) in crisis_periods.items():
    period = df[(df['Date'] >= start) & (df['Date'] <= end)]
    if len(period) > 0:
        max_vix = period['VIX_Close'].max()
        days_over_40 = (period['VIX_Close'] > 40).sum()
        if max_vix > 30 or days_over_40 > 0:
            print(f"\n{name} ({start} to {end}):")
            print(f"  Max VIX: {max_vix:.2f}")
            print(f"  Days VIX > 40: {days_over_40}")

