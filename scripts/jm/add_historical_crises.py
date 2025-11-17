"""
Add more historical crisis tasks by lowering minimum duration threshold.

Currently using: 30 days minimum
This script: Try 20 days to capture more short crises

Target crises to capture:
- 1987 Black Monday (Oct 1987)
- 1997 Asian Crisis (Oct-Nov 1997)
- 2010 Flash Crash (May 2010)
- 2015 China Crisis (Aug 2015)
- 2018 Q4 Crash (Dec 2018)
"""

import pandas as pd
import json
from pathlib import Path

def analyze_potential_crises():
    """Check what crises we're missing with current threshold"""
    
    # Load jump regime labels
    regime_labels = pd.read_csv('data/jm/jump_regime_labels.csv', parse_dates=['Date'])
    
    # Find all JR2 segments (even short ones)
    jr2_segments = []
    in_jr2 = False
    start_date = None
    
    for idx, row in regime_labels.iterrows():
        if row['jump_regime'] == 2 and not in_jr2:
            # Start of JR2 segment
            in_jr2 = True
            start_date = row['Date']
        elif row['jump_regime'] != 2 and in_jr2:
            # End of JR2 segment
            in_jr2 = False
            end_date = regime_labels.iloc[idx-1]['Date']
            duration = (end_date - start_date).days
            jr2_segments.append({
                'start': start_date,
                'end': end_date,
                'duration': duration
            })
    
    # Sort by duration
    jr2_segments.sort(key=lambda x: x['duration'], reverse=True)
    
    print("=" * 60)
    print("ALL JR2 CRISIS PERIODS (by duration)")
    print("=" * 60)
    
    print("\n✅ Currently Used (≥30 days):")
    for seg in jr2_segments:
        if seg['duration'] >= 30:
            print(f"  {seg['start'].date()} to {seg['end'].date()}: {seg['duration']}d")
    
    print("\n⚠️  Excluded (20-29 days) - COULD ADD:")
    for seg in jr2_segments:
        if 20 <= seg['duration'] < 30:
            print(f"  {seg['start'].date()} to {seg['end'].date()}: {seg['duration']}d")
    
    print("\n❌ Too Short (<20 days) - Skip:")
    for seg in jr2_segments:
        if seg['duration'] < 20:
            print(f"  {seg['start'].date()} to {seg['end'].date()}: {seg['duration']}d")
    
    print(f"\n📊 Summary:")
    print(f"  Current tasks (≥30d): {len([s for s in jr2_segments if s['duration'] >= 30])}")
    print(f"  Potential new (20-29d): {len([s for s in jr2_segments if 20 <= s['duration'] < 30])}")
    print(f"  Total if lowered: {len([s for s in jr2_segments if s['duration'] >= 20])}")
    
    return jr2_segments

def estimate_impact():
    """Estimate if adding more tasks would help"""
    
    print("\n" + "=" * 60)
    print("IMPACT ANALYSIS")
    print("=" * 60)
    
    # Historical crises we WANT to capture
    target_crises = {
        '1987-10': 'Black Monday (fastest crash)',
        '1997-10': 'Asian Crisis (contagion)',
        '2010-05': 'Flash Crash (intraday panic)',
        '2015-08': 'China Crisis (circuit breakers)',
        '2018-12': 'Q4 Crash (correction)'
    }
    
    print("\n🎯 Target Historical Crises:")
    for period, name in target_crises.items():
        print(f"  {period}: {name}")
    
    print("\n💡 Recommendation:")
    print("  1. Lower threshold: 30 days → 20 days")
    print("  2. Expected gain: 5 tasks → 8-10 tasks")
    print("  3. Benefit: More diverse crisis patterns")
    print("  4. Risk: Shorter tasks = less data per task")
    
    print("\n⚠️  Trade-off:")
    print("  - MORE tasks = better generalization")
    print("  - SHORTER tasks = noisier adaptation")
    print("  - Sweet spot: 20-25 days minimum")

if __name__ == '__main__':
    segments = analyze_potential_crises()
    estimate_impact()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Review the 20-29 day segments above")
    print("2. Check if they include 1987, 2010, 2015, 2018")
    print("3. If YES: Regenerate tasks with 20-day minimum")
    print("4. If NO: These crises might be <20 days (too volatile)")
    print("\nTo regenerate:")
    print("  python scripts/jm/generate_segment_based_tasks.py --min_duration 20")
