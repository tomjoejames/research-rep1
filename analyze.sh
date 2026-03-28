# python experiments/e4_cross_device.py
python scripts/analyze_results.py results/e5_timeseries_device1_20260325_042207.json
python scripts/analyze_results.py results/e5_timeseries_device2_20260325_141500.json

git checkout main
git add .
git commit -m 'results- round 1'
git push origin -u main