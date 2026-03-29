python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./experiments/setup_models.sh
python experiments/e1_baseline.py --device device3
python experiments/e2_quantization.py --device device3
python experiments/e3_agent_overhead.py --device device3
python experiments/e5_memory.py --device device3
python experiments/e6_coldwarm.py --device device3
git checkout -b results-
git add results/
git commit -m 'result 1'
git push origin results-device3