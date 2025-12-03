# CS182 Project — In-Context Learning Across Architectures

## 🔭 Project Vision
Sequence-model experiments and dataset builders to study whether **self-attention is required for In-Context Learning (ICL)**. We compare minimal sequence architectures (RNN/LSTM/SSM-style) to Transformers at similar scale using:

- **Amazon Reviews sentiment prediction** (natural text ICL signal)
- **Large synthetic recursive sequences** (noisy numeric + alphabet encoded)
- **5–8 example contexts per sample** (few-shot supervision without parameter updates)

## 🧠 Core Hypotheses
1. ICL is a property of **general sequence modeling**, not uniquely self-attention.
2. **Structured recursion + controlled noise** leads models to learn estimators instead of memorizing canonical sequences.
3. Architecture shapes **how an estimator emerges**, not whether ICL works at all.

## 🧪 Key Datasets
We provide both **natural text** and **controlled synthetic** data:

- `amazon_review.csv` — raw Amazon reviews for sentiment experiments
- `amazon_reviews_sentiment.csv` — cleaned reviews with 5–8 context examples per sample for rating prediction
- `noisy_fibonacci.csv/json` — numeric noisy Fibonacci-style recursion with ε ∈ {-2, -1, 0, 1, 2}
- `letter_fibonacci.csv` — digit→alphabet encoded Fibonacci (0→a, 1→b, …, 9→j)
- `typo_generator.csv` — 100k noise/clean column dataset for typo-correction ICL

## 📁 Repository Layout

cs182-project/
├── data_generator.py # generates synthetic recursive sequences + noise injection
├── data_parse.py # review + synthetic data preprocessing pipeline
├── rnn_gpu.py # RNN experiments with GPU training
├── rnn_reg_gpu.py # RNN regression variant (GPU)
├── transformer_gpu.py # Transformer baseline (GPU)
├── *.csv / *.json # generated or parsed datasets
└── README.md # this documentation


## ⚙️ Environment Setup
Clone and prepare:

```bash
git clone cs182-project
cd cs182-project
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy torch tqdm
```

## ▶️ Usage
Generate synthetic recursive data
```
python data_generator.py --rows 500000 --noise 2 --min_seed -100 --max_seed 100 --out noisy_fibonacci.csv
```
Parse Amazon review or synthetic data
```
python data_parse.py --input amazon_review.csv --output amazon_reviews_processed.csv
```

Train baseline models
```
python transformer_gpu.py --data amazon_reviews_processed.csv --epochs 10 --batch_size 64
python rnn_gpu.py --data amazon_reviews_processed.csv --epochs 10 --batch_size 64
python rnn_reg_gpu.py --data noisy_fibonacci.csv --epochs 15 --batch_size 128
```

## ⚠️ Limitations

Synthetic data is distribution-controlled, not realism-matched

Few-shot estimator discovery may need manual logging & analysis harness

GPU scripts require a compatible backend like PyTorch

## 🤝 Authors

Research and engineering collaboration by the project team:

Rohan Gulati,
Alena Chao,
Natalie Wei,
Andrew Choy,
Minjune Kim

## 📚 Background References

Scaling and discovery of few-shot ICL was first shown in OpenAI via GPT-3 (Brown et al. 2020)

Estimator emergence in sequence models has been explored using non-attention architectures like RNNs, LSTMs, and SSM families

Controlled recursive-noise datasets are used to avoid memorization while preserving additive structure