# 5G Network Performance Analysis — Study Guide

Everything in this project, explained simply, in the order the program runs.
Read it once slowly, then practise saying the **"Say this out loud"** boxes.

---

## Part 1 — The 30-second answer

If your teacher says *"So what is your project?"*, say this:

> "It's a website that studies 5G mobile network data. You give it a CSV file of
> network readings and it does four things: cleans the data, draws graphs to
> explain it, tests your real internet speed right now, and uses machine
> learning to predict how fast the network will be. It's written in Python
> using a library called Streamlit."

That's it. Don't say more until they ask.

---

## Part 2 — Eight words you must be able to define

If you know only these eight, you can survive any question. Learn them first.

**1. Streamlit**
A Python library that turns a Python script into a web page automatically.
Normally a website needs HTML, CSS and JavaScript. With Streamlit you write
`st.title("Hello")` in Python and a heading appears in the browser. That's why
our whole website is just `.py` files and no HTML anywhere.

**2. DataFrame**
A table in Python — rows and columns, like an Excel sheet. It comes from the
`pandas` library. We always call ours `df`, short for dataframe.

**3. CSV**
"Comma Separated Values" — a plain text file where each line is a row and
commas separate the columns. Our dataset `5g_network_data.csv` has 50,000 rows
and 21 columns.

**4. session_state**
Streamlit's memory box. Normally when you click something, Streamlit re-runs
the whole script from the top and forgets everything. `st.session_state` is a
dictionary that survives between clicks and between pages. We store our cleaned
data there so all 8 pages can use the same data without reloading the file.

**5. Numeric vs categorical**
Numeric = numbers you can do maths on (speed, latency, temperature).
Categorical = labels/text (Airtel, Jio, iPhone 14, "High"). They need
completely different handling, which is why our code keeps splitting columns
into these two groups.

**6. One-hot encoding**
Machine learning models only understand numbers, not the word "Airtel". One-hot
encoding turns one text column into several 0/1 columns:

| Carrier | → | Airtel | Jio | AT&T |
|---|---|---|---|---|
| Airtel | | 1 | 0 | 0 |
| Jio | | 0 | 1 | 0 |

**Why not just number them 1, 2, 3?** Because then the model would think
Jio (2) is "twice" Airtel (1), which is meaningless. One-hot avoids that fake
ordering.

**7. Train/test split**
We give the model 80% of the data to learn from, and hide 20% from it. Then we
test it on the hidden 20%.

*Why?* Because testing a model on the same data it studied is like giving a
student the exam paper the night before — you learn nothing about whether they
actually understood.

**8. R² (R-squared)**
A score for how good the model is. 1.0 = perfect. 0 = no better than just
guessing the average every time. Negative = worse than guessing the average.

---

## Part 3 — The flow (draw this on the board)

```
        CSV file
           │
           ▼
   ┌──────────────┐
   │   app.py     │   loads it, cleans it,
   │  (Home page) │   puts it in session_state
   └──────┬───────┘
          │
    ┌─────┴──────────────────────────────┐
    │        the 8 pages read it         │
    ├────────────────────────────────────┤
    │ 1  Network Performance   graphs    │
    │ 2  Live Speed Test       real test │
    │ 3  EDA & Statistics      summary   │
    │ 4  Correlation           (partner) │
    │ 5  ML Prediction    ◄─── trains    │
    │ 6  Model Comparison ◄─── reads 5   │
    │ 7  Actual vs Predicted   (partner) │
    │ 8  Location Analysis     by city   │
    └────────────────────────────────────┘

  All of them use helpers from the utils/ folder.
```

**The key point to make:** pages 6 and 7 do not train anything themselves. They
just read the results that page 5 saved into `session_state`. That's why if you
open page 6 first, it says *"Train models on the ML Prediction page first."*

---

## Part 4 — The `utils/` folder (the helper files)

`utils` is short for "utilities" — reusable code. These files have **no web page
of their own**. They are the toolbox that the pages borrow from.

> **Why put code here instead of inside the pages?**
> So we don't repeat ourselves. Five pages need to load the data. If that code
> lived inside each page we'd have written it five times, and fixing a bug would
> mean fixing it in five places. This is called *separation of concerns* —
> the pages handle looks, the utils handle logic.

---

### 4.1 — `utils/__init__.py`

**This file is completely empty.** That is not a mistake.

Python only treats a folder as an importable "package" if it contains a file
named `__init__.py`. Its mere existence is what allows us to write
`from utils.data_utils import load_csv`. Without it, Python would say
"no module named utils".

> **Say this out loud:** "It's an empty marker file. It tells Python that
> `utils` is a package and not just an ordinary folder, so we can import from it."

---

### 4.2 — `utils/data_utils.py` — loading and cleaning

Four functions.

**`load_csv(file_or_path)`**
```python
df = pd.read_csv(file_or_path)
df.columns = [c.strip() for c in df.columns]
```
Reads the CSV into a dataframe. Then `.strip()` removes accidental spaces from
every column name — because if a heading is written as `" Latency "` with spaces,
then `df["Latency"]` would fail and crash the app. One line that prevents a
very annoying bug.

Note it accepts *either* an uploaded file *or* a file path, which is why the
same function works for both the upload button and the built-in sample dataset.

**`validate_dataset(df)`**
Checks three things and returns a list of problems:
- Is the table empty?
- Does it have fewer than 10 rows? (too small to analyse)
- Are two columns named the same thing? (would confuse everything)

It **returns** the problems instead of stopping the app, so the user sees a
yellow warning but can still continue.

**`dataset_summary(df)`**
Counts rows, columns, missing values, duplicate rows, and lists which columns
are numeric and which are categorical. These are the four boxes at the top of
the home page.

**`preprocess(df)` — the important one**

This is the actual cleaning. Three steps:

1. `df.drop_duplicates()` — removes rows that are 100% identical to another row.
2. For every **numeric** column with blanks → fill with the **median**.
3. For every **categorical** column with blanks → fill with the **mode**.

> **Median** = the middle value when you sort them.
> **Mode** = the most frequently occurring value.

**The question your teacher will definitely ask: "Why median and not mean?"**

Because the mean gets dragged around by extreme values. If four readings are
50, 55, 60 and 9000 Mbps, the mean is 2291 — a number that describes none of
the readings. The median is 57. Network data is full of these weird spikes, so
median is the safe choice.

For text columns you can't take a median at all, so we use the most common
value instead.

**`get_dataframe()`**
```python
if st.session_state.get("df_clean") is not None:
    return st.session_state["df_clean"]
return st.session_state.get("df")
```
Every page calls this one function to fetch the data. It prefers the cleaned
version, falls back to the raw version, and returns `None` if nothing is loaded
yet — which is what makes the "Load a dataset on the Home page first" warning
work on every page.

---

### 4.3 — `utils/quality_utils.py` — the quality score

**The problem this solves:** after a speed test you have five separate numbers.
Is 180 Mbps download with 40 ms ping better or worse than 90 Mbps with 8 ms
ping? Hard to say. So we squash all five into **one score out of 100** plus a
label like "Excellent".

**`_normalize(value, best, worst)`**
```python
score = (value - worst) / (best - worst) * 100
```
Converts any measurement onto a 0–100 scale. Feed it your value, plus what
counts as best and worst, and it tells you where you sit as a percentage.
`np.clip(score, 0, 100)` then forces the answer to stay inside 0–100 even if
your value goes beyond the reference range.

**The clever bit** — for ping, jitter and packet loss, *lower is better*. So we
pass `best=5, worst=150` — the "best" number is smaller than the "worst" number.
The subtraction then flips direction automatically, and low ping correctly
scores high. No separate formula needed.

Also notice `if best == worst: return 100.0`. That's there to prevent a
**divide-by-zero crash**, since the denominator `(best - worst)` would be 0.
Mentioning this shows you thought about edge cases.

**`compute_quality_score(...)` — the weights**

| Measurement | Weight | Why |
|---|---|---|
| Download speed | 40% | What you feel most — streaming, downloads |
| Upload speed | 20% | Matters for video calls, uploading |
| Ping | 20% | Matters for gaming and responsiveness |
| Jitter | 10% | Causes stuttering in calls |
| Packet loss | 10% | Causes freezing |

The weights add up to 1.0 (40+20+20+10+10 = 100%). Download gets the biggest
share because it affects the most common activities.

**`classify_quality(score)`**
Walks down a list of thresholds and returns the first label the score reaches:
90+ Excellent, 75+ Very Good, 60+ Good, 40+ Average, below that Poor.

**Be ready for: "Where did these weights come from?"**
Honest answer: *"We chose them ourselves based on what affects a user most, and
we wrote them in one place in the code so they can be tuned. They are not an
official telecom standard."* Saying that is much stronger than pretending they
came from a specification.

---

### 4.4 — `utils/speedtest_utils.py` — the real speed test

This is the only part of the project that touches the actual internet.

**`run_speed_test()`**
```python
client = speedtest.Speedtest(secure=True)
client.get_best_server()
download_bps = client.download()
upload_bps   = client.upload()
```
Uses the `speedtest-cli` library — the same Speedtest.net service you use on
your phone. `get_best_server()` finds the nearest server first, because testing
against a server in another country would measure the distance, not your
connection. Then it downloads and uploads test data and measures how long it
took.

The result comes back in **bits per second**, so we divide by 1,000,000 to get
Mbps.

**`_ping_stats()` — latency, jitter and packet loss**

We don't use the speedtest library for these. Instead we run the computer's own
`ping` command and read its output:

```python
is_windows = platform.system().lower() == "windows"
cmd = ["ping", "-n" if is_windows else "-c", str(count), host]
```

**Two things to point out here, both good marks:**

*Why the OS `ping` command instead of a Python ping library?* Because sending
raw network packets from Python needs administrator/root permission. Using the
built-in command means the app runs on any normal user account.

*Why the `is_windows` check?* Windows writes the flag as `-n 8`, but Mac and
Linux write it as `-c 8`. This one line makes the project cross-platform.

Then a **regular expression** pulls the times out of the text output:
```python
times = [float(m) for m in re.findall(r"time[=<]\s*([\d.]+)\s*ms", output)]
```
It hunts for the pattern `time=12.4 ms` and collects every number it finds.

From those numbers:
- **ping** = `statistics.mean(times)` — the average
- **jitter** = `statistics.pstdev(times)` — the standard deviation, i.e. how
  much the ping *varies*. Steady ping = low jitter = good.
- **packet loss** = what percentage of pings never came back

> **Explaining jitter simply:** "Ping is how long a message takes. Jitter is
> whether that time keeps changing. A steady 40 ms is better for a video call
> than a ping that jumps between 10 and 90 ms, even though the average is
> similar."

**`save_result()` and `load_history()`**
Every test gets appended as one row to `data/speed_test_history.csv`. `mode="a"`
means append (add to the end) rather than overwrite, and the header row is only
written the very first time. This is what lets page 2 show a graph of your
speed over days.

---

### 4.5 — `utils/ml_utils.py` — the machine learning engine

The biggest and most important file. Expect the most questions here.

**`MODEL_FACTORY` — the four models**

| Model | How it works, in one line |
|---|---|
| **Linear Regression** | Draws the best-fit straight line through the data |
| **Decision Tree** | A flowchart of yes/no questions: "is signal < -90? then..." |
| **Random Forest** | Builds 100 different trees and averages their answers |
| **Gradient Boosting** | Builds trees one after another, each one fixing the previous one's mistakes |

*Why a Random Forest beats a single tree:* one tree can memorise the training
data too closely (this is called **overfitting**). A hundred slightly different
trees voting together cancel out each other's mistakes.

It's called a "factory" because it stores **instructions for building** a model
(`lambda: LinearRegression()`), not a built model. We need a fresh, untrained
model each time we train.

**`PREDICTION_TARGETS` and `TARGET_LIKE`**

The **target** is what we're predicting. The **features** are what we predict
*from*.

`TARGET_LIKE` is a list of columns we refuse to use as features: Download Speed,
Upload Speed, Latency, Jitter, Ping to Google, Video Streaming Quality, Dropped
Connection.

**Why? This is the single best point in your whole presentation:**

> If we're predicting Download Speed and we let the model see "Video Streaming
> Quality" as an input, that's cheating — video quality is *caused by* download
> speed. The model would get a great score without learning anything real. This
> problem is called **data leakage**. So we only feed it *causes* (signal
> strength, carrier, band, congestion, time of day), never other *effects*.

**`prepare_features(df, feature_cols)`**

Two jobs.

*Job 1 — the timestamp.* A model can't use the text `"2025-05-28 06:59:51"`.
So we split it into three useful numbers and throw the original away:
```python
X["Hour"]        = ts.dt.hour        # 0-23
X["Day of Week"] = ts.dt.dayofweek   # 0=Monday
X["Month"]       = ts.dt.month
X = X.drop(columns=["Timestamp"])
```
Now the model can learn things like "the network is slower at 9 PM" or
"weekends are different from weekdays" — which the raw text could never express.

*Job 2* — converts `True`/`False` columns into the text `"True"`/`"False"`,
because some versions of scikit-learn handle raw booleans inconsistently.

**`make_preprocessor(X)` — two parallel assembly lines**

Numeric and categorical columns need different treatment, so we build two
pipelines and run them side by side using a `ColumnTransformer`:

```
numeric columns    →  fill blanks with median  →  scale
categorical columns →  fill blanks with mode   →  one-hot encode
```

*What is scaling (`StandardScaler`)?* Signal strength is around −90 and data
usage is around 500. Without scaling, a model like Linear Regression assumes the
bigger numbers are more important just because they're bigger. Scaling rewrites
every column onto the same footing so they compete fairly.

*What is `handle_unknown="ignore"`?* If the model was trained on Airtel/Jio/AT&T
and later meets "Vi", it will output all zeros instead of crashing.

**`train_and_evaluate(...)` — the main function**

Step by step:
1. Build the feature table with `prepare_features`
2. Convert the target to numbers; drop any row where the target is missing
   (you can't learn from an answer that isn't there)
3. `train_test_split(X, y, test_size=0.2, random_state=42)` — 80/20 split
4. For each of the four models: build a fresh pipeline, `.fit()` it on the
   training data, `.predict()` on the test data
5. Score it and store everything in a dictionary

**"What is `random_state=42`?"** — a fixed seed for the random shuffling. It
makes the split identical every time you run it, so your results are
**reproducible**. Without it, your R² would wobble slightly on every run and
you couldn't fairly compare models. (42 is just a programmers' joke number;
any number works.)

**The three error measurements**

| Metric | Meaning | Better when |
|---|---|---|
| **MAE** | Mean Absolute Error — average size of the mistake, in Mbps | Lower |
| **RMSE** | Root Mean Squared Error — like MAE but punishes big mistakes harder, because errors get squared first | Lower |
| **R²** | Fraction of the pattern the model captured, 0 to 1 | Higher |

*MAE vs RMSE:* if MAE is 20 but RMSE is 60, that gap tells you the model is
usually close but occasionally very wrong. Squaring makes one huge error count
far more than several small ones.

**`feature_importance(model_bundle)`**
Tree models can report which inputs they relied on most. This returns the top 20
so page 5 can draw the bar chart. Linear Regression has no
`feature_importances_` attribute, so the function checks with `hasattr` and
returns an empty table instead of crashing.

---

## Part 5 — `app.py` (the Home page)

The file Streamlit runs first: `streamlit run app.py`.

**The path fix at the top**
```python
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```
This tells Python where the project folder is, so `from utils...` works no
matter which folder you launched the app from. Without it you can get
"no module named utils" errors.

**`st.set_page_config(...)`** — sets the browser tab title, the tab icon, and
`layout="wide"` so charts use the full screen width. **It must be the first
Streamlit command in the file** — that's a Streamlit rule.

**The three-way data fallback**
```python
if uploaded is not None:      # 1. user's own file
elif DEFAULT_PATH.exists():   # 2. our sample dataset
else:                         # 3. nothing → stop
```
This is why the app already shows graphs the moment you open it, instead of an
empty screen. Good for a demo.

**`st.stop()`** — halts the script right there. Everything below simply never
runs. We use it whenever there's no data, so we never try to draw a chart of
nothing.

**Publishing the data**
```python
st.session_state["df"] = df
st.session_state["df_clean"] = df
```
This one line is what connects the home page to all 8 other pages.

The rest is display: four metric boxes, a 50-row preview, `df.describe()`
statistics, a line chart of any numeric column you pick, and a download button
for the cleaned data.

> `.head(50)` is deliberate — showing all 50,000 rows would make the browser
> crawl.

---

## Part 6 — The eight pages

### Page 1 — Network Performance

Pick a numeric column and see two charts of it: a **line chart** (how it changes
across records) and a **histogram** (which values are most common). You can
optionally group by a categorical column like Carrier, which colours the charts
by group.

If you choose a group, it also shows a table of average values per group —
`df.groupby(filter_col)[numeric_cols].mean()`. That's how you'd answer "which
carrier is fastest on average?"

**The dynamic part to highlight:** we never hard-code column names. The page
asks the dataframe what columns exist (`df.select_dtypes`) and builds the
dropdowns from that. Upload a totally different CSV and the page still works.

---

### Page 2 — Live Speed Test

The demo page. Click **Start Test**, it calls `run_speed_test()`, computes the
quality score, saves it to history, and shows five metric boxes plus the
"Excellent / Good / Poor" verdict.

Notice the test is wrapped in `try / except`. If there's no internet, the app
shows a red error message instead of crashing. Say this — error handling gets
marks.

`st.spinner(...)` shows the "Running speed test…" message so the user knows the
app is working and not frozen.

Below that, the full history table and a line graph of download/upload over time.

⚠️ **Practical warning:** this needs a working internet connection and school
Wi-Fi sometimes blocks the speedtest servers. **Test it on the actual demo
machine beforehand,** and take a screenshot as backup.

---

### Page 3 — EDA & Statistics

**EDA = Exploratory Data Analysis.** The step where you look around and
understand the data *before* building any model.

Shows:
- `df.describe().T` — count, mean, standard deviation, min, max and the
  quartiles for every numeric column
- a **histogram with a box plot** on the margin, for any column you pick
- a separate **box plot** to spot outliers
- a list of missing values per column, or a green "No missing values" message
- the duplicate row count

**What a box plot shows:** the box holds the middle 50% of the data, the line
inside is the median, and the dots floating outside are **outliers** — unusually
high or low readings.

**Why EDA matters (say this):** "You can't build a good model on data you don't
understand. EDA is how we found out our dataset had problems."

---

### Page 4 — Correlation Analysis

*(Your partner's page — just know what it is.)* It shows how strongly pairs of
numeric columns move together, as a colour-coded heatmap, plus a scatter plot
explorer.

---

### Page 5 — ML Prediction

The heart of the project. Four steps on one page:

**1. Choose what to predict.** Default is Download Speed.

**2. Choose the inputs.** Pre-filled by `default_feature_columns()`, which
automatically excludes the leakage columns. You can add or remove any of them.

**3. Click Train.** Runs all four models and shows the best one, its R², its
RMSE, a comparison table, and the feature-importance bar chart.

**4. Make a prediction.** A form appears with one input per feature. The code
builds these widgets automatically based on each column's type:
- boolean → True/False dropdown
- numeric → a number box limited to that column's real min and max
- text → a dropdown of the actual values in the data

Then it predicts a single value.

**Two things to be honest about:**

⚠️ Step 4 currently crashes. The model was trained on Hour/Day/Month, but the
form sends the raw Timestamp, so it errors with *"columns are missing: {'Hour',
'Day of Week', 'Month'}"*. It's a small fix (send the input through
`prepare_features` first) but fix it before you demo.

⚠️ The results also store in `session_state["ml_results"]`, which is exactly how
pages 6 and 7 get their data.

---

### Page 6 — Model Comparison

Reads the results page 5 saved. Shows MAE, MSE, RMSE and R² for all four models
in one table sorted by R², announces the winner, and draws two bar charts.

**It trains nothing.** If you open it first you get a warning telling you to go
to page 5. That's deliberate design — no point retraining four models when the
answers already exist in memory.

---

### Page 7 — Actual vs Predicted

*(Your partner's page.)* A scatter plot of what really happened versus what the
model guessed, plus a histogram of the residuals (the errors).

---

### Page 8 — Location Analysis

Compares network performance by place.

The smart part: it doesn't assume a column called "Location" exists. It
**searches** for any column whose name contains location, city, region, cell,
site or area:
```python
if any(k in c.lower() for k in ["location", "city", "region", ...])
```
It also looks for latitude and longitude columns. If it finds none of these, it
shows a polite message instead of crashing.

If it finds a location column → average metrics per location, as a table and a
grouped bar chart. If it finds lat/long → an actual map with a dot per reading,
coloured by whichever metric you choose.

---

## Part 7 — The honest problem, and how to handle it

You will probably be asked why your R² is near zero. **Do not panic and do not
bluff.** Here is the truth, and the truth actually sounds impressive:

> "Our sample dataset is randomly generated, so there is no real relationship
> hidden in it to learn. Every column correlates with download speed at less
> than 0.01, and signal strength has no effect on speed at all — which is
> physically impossible in real network data. All four models score around zero
> R², which is exactly the correct result for random data. We found this using
> our own correlation and EDA pages, which is what those pages are for. To fix
> it we need either a real-world dataset or a data generator where speed
> genuinely depends on signal strength, congestion and band."

That answer shows you understand your own tools, understand what R² means, and
can diagnose a problem. That's worth far more than a fake high score.

---

## Part 8 — Rapid-fire practice questions

Cover the answers and test yourself.

**Why Streamlit and not Flask or Django?**
Streamlit is built for data apps. Charts, tables and widgets are one line each,
with no HTML. Flask would need separate template files and much more code.

**Why split code into `utils/` and `pages/`?**
Separation of concerns. Pages handle the interface, utils handle the logic.
Shared code is written once, so a fix applies everywhere.

**What happens when you click a button in Streamlit?**
The entire script re-runs from the top. That's why `session_state` is needed —
it's the only thing that survives the re-run.

**Why fill missing values instead of deleting those rows?**
Deleting throws away all the other good columns in that row. With 21 columns,
one blank cell would cost you 20 useful values.

**Which model is best and why?**
Whichever the comparison page shows — normally Random Forest or Gradient
Boosting, because they can learn curved and combined relationships, while Linear
Regression can only fit a straight line.

**What's the difference between MAE and RMSE?**
Both measure error. RMSE squares the errors first, so it punishes a few big
mistakes much more heavily. If RMSE is much larger than MAE, your model has some
badly wrong predictions.

**What is overfitting?**
When a model memorises the training data instead of learning the pattern. It
scores brilliantly on data it has seen and badly on new data. The train/test
split is how we detect it.

**Why is `Dropped Connection` not used as the target?**
Two reasons: it's roughly a random 50/50 in this dataset so there's nothing to
learn, and it's a yes/no outcome, which needs classification models, whereas all
four of our models are regression models for continuous numbers.

**Where is the data stored?**
`data/5g_network_data.csv` for the dataset, `data/speed_test_history.csv` for
speed test results. There's no database — for a project this size, CSV files are
simpler and easier to inspect.

**How would you improve this project?**
Add caching so models don't retrain on every click, add a classification model
for dropped connections, write automated tests, and most importantly use a real
dataset instead of the random one.

---

## Part 9 — Your three-minute demo script

1. **Home page** — "This is the dataset: 50,000 rows, 21 columns, and it tells
   us straight away how many values are missing or duplicated."
2. **Page 3, EDA** — "Here we explore it. This box plot shows the outliers."
3. **Page 1** — "We can chart any metric and group it by carrier to compare."
4. **Page 8** — "Performance broken down by city."
5. **Page 2** — "And this tests the real internet connection live." *(Have a
   screenshot ready in case the Wi-Fi blocks it.)*
6. **Page 5** — "Now the machine learning. We choose what to predict, it trains
   four models, and it tells us which one performed best and which inputs
   mattered most."
7. **Page 6** — "And here's the side-by-side comparison of all four."
8. **The honest bit** — deliver the Part 7 answer about the dataset before
   anyone has to ask you.

Good luck. Practise saying it out loud at least twice — reading it silently
feels like you know it, but speaking it is a different skill.
