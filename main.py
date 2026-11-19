import os
import re
import json
import webbrowser
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from scipy.stats import zscore, norm, uniform, binom, pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
fake = Faker()
fake.seed_instance(SEED)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# етап 1: генериране и зареждане на данните
def  generate_data():
    print("--- ЕТАП 1: ГЕНЕРИРАНЕ НА ДАННИ ---")
    cities = [fake.unique.city() for _ in range(6)]
    fake.unique.clear()

    df_customers = pd.DataFrame([{
        "CustomerID": f"CUST_{i:04d}",
        "Name": fake.name(),
        "Age": int(np.random.randint(18, 70)),
        "City": fake.random_element(cities),
        "RegistrationDate": fake.date_between(start_date="-2y", end_date="-1y"),
        "Category": np.random.choice(["Standard", "Silver", "Gold", "VIP"], p=[0.5, 0.25, 0.15, 0.10])
    } for i in range(1, 121)])

    products_data = []
    for i in range(1, 26):
        cost = round(np.random.uniform(10.0, 500.0), 2)
        products_data.append({
            "ProductID": f"PROD_{i:03d}",
            "ProductName": f"Product_{i:03d}_{fake.word().capitalize()}",
            "Category": fake.random_element(["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]),
            "UnitPrice": round(cost * np.random.uniform(1.25, 1.80), 2),
            "Cost": cost
        })
    df_products = pd.DataFrame(products_data)

    date_range = pd.date_range(start="2025-11-01", end="2026-10-31", freq="D")
    weights = np.where(date_range.month.isin([11, 12]), 1.8, 1.0)
    sampled_dates = np.random.choice(date_range, size=1200, p=weights / weights.sum())

    df_orders = pd.DataFrame([{
        "OrderID": f"ORD_{i:05d}",
        "CustomerID": np.random.choice(df_customers["CustomerID"]),
        "OrderDate": pd.Timestamp(sampled_dates[i - 1]).strftime("%Y-%m-%d"),
        "Status": np.random.choice(["Completed", "Processing", "Cancelled", "Returned"], p=[0.82, 0.08, 0.06, 0.04]),
        "PaymentMethod": fake.random_element(["Credit Card", "Cash on Delivery", "Bank Transfer", "Digital Wallet"])
    } for i in range(1, 1201)])

    price_map = dict(zip(df_products["ProductID"], df_products["UnitPrice"]))
    details_data, det_id = [], 1
    for oid in df_orders["OrderID"]:
        for pid in np.random.choice(df_products["ProductID"], size=np.random.randint(1, 5), replace=False):
            qty = int(np.random.randint(1, 6))
            details_data.append({
                "DetailID": f"DET_{det_id:06d}", "OrderID": oid, "ProductID": pid,
                "Quantity": qty, "UnitPrice": price_map[pid], "LineTotal": round(qty * price_map[pid], 2)
            })
            det_id += 1
    df_order_details = pd.DataFrame(details_data)

    phrases = {
        "pos": ([4, 5], ["Great product, fast delivery and amazing quality!",
                         "Love this item, works perfectly and worth every penny.",
                         "Excellent customer service and very reliable packaging.",
                         "Super satisfied with my purchase, will buy again!"]),
        "neu": ([3], ["The product is okay, nothing special but does the job.",
                      "Average quality, delivery took a bit longer than expected.",
                      "It matches the description, fair price for what you get."]),
        "neg": ([1, 2], ["Terrible quality, broke after two days. Very disappointed!",
                         "Worst purchase ever, late shipping and poor support.",
                         "Not worth the money, completely different from the photos."])
    }
    reviews_data = []
    for idx, (_, row) in enumerate(df_orders.sample(n=350, random_state=SEED).iterrows(), 1):
        s_type = np.random.choice(["pos", "neu", "neg"], p=[0.65, 0.20, 0.15])
        ratings, pool = phrases[s_type]
        reviews_data.append({
            "ReviewID": f"REV_{idx:04d}", "OrderID": row["OrderID"], "CustomerID": row["CustomerID"],
            "ReviewDate": (pd.to_datetime(row["OrderDate"]) + pd.Timedelta(days=int(np.random.randint(1, 15)))).strftime("%Y-%m-%d"),
            "ReviewText": f"{fake.random_element(pool)} {fake.sentence()}",
            "Rating": int(np.random.choice(ratings))
        })
    df_reviews = pd.DataFrame(reviews_data)

    df_customers.loc[df_customers.sample(frac=0.03, random_state=SEED).index, "Age"] = np.nan
    df_order_details.loc[df_order_details.sample(frac=0.015, random_state=SEED).index, "Quantity"] = np.nan

    dirty_city = df_customers.sample(frac=0.05, random_state=SEED).index
    df_customers.loc[dirty_city, "City"] = df_customers.loc[dirty_city, "City"].apply(lambda x: f"  {x.lower()} ")

    dirty_date = df_orders.sample(frac=0.03, random_state=SEED).index
    df_orders.loc[dirty_date, "OrderDate"] = pd.to_datetime(df_orders.loc[dirty_date, "OrderDate"]).dt.strftime("%d.%m.%Y")

    out_idx = df_order_details.sample(n=8, random_state=SEED).index
    df_order_details.loc[out_idx, "Quantity"] = df_order_details.loc[out_idx, "Quantity"].fillna(1) * 25
    df_order_details.loc[out_idx, "LineTotal"] = df_order_details.loc[out_idx, "Quantity"] * df_order_details.loc[out_idx, "UnitPrice"]

    df_orders = pd.concat([df_orders, df_orders.sample(frac=0.01, random_state=SEED)], ignore_index=True)
    df_reviews = pd.concat([df_reviews, df_reviews.sample(frac=0.02, random_state=SEED)], ignore_index=True)

    for name, df in [("customers", df_customers), ("products", df_products), ("orders", df_orders),
                     ("order_details", df_order_details), ("reviews", df_reviews)]:
        df.to_csv(f"{DATA_DIR}/{name}.csv", index=False, encoding="utf-8-sig")
    print("CSV файловете са записани в 'data/'.\n")


def load_and_inspect_data():
    print("--- ЕТАП 1: ЗАРЕЖДАНЕ И ПЪРВОНАЧАЛНА ПРОВЕРКА (EDA) ---")
    dfs = {name: pd.read_csv(f"{DATA_DIR}/{name.lower()}.csv")
           for name in ["Customers", "Products", "Orders", "Order_Details", "Reviews"]}
    for name, df in dfs.items():
        print(f"\n=== {name} | Shape: {df.shape} | Duplicates: {df.duplicated().sum()} ===")
        print(df.head(3))
        print("Липсващи стойности:\n", df.isna().sum()[df.isna().sum() > 0])
        print(df.describe())
    return dfs["Customers"], dfs["Products"], dfs["Orders"], dfs["Order_Details"], dfs["Reviews"]


# етап 2: предварителна обработка и почистване
def preprocess_data(customers, products, orders, order_details, reviews):
    print("\n--- ЕТАП 2: ПРЕДВАРИТЕЛНА ОБРАБОТКА ---")
    before = {"Orders_rows": len(orders), "OrderDetails_missing": order_details["Quantity"].isna().sum(),
              "Max_Qty": order_details["Quantity"].max(), "Reviews_rows": len(reviews)}

    customers = customers.drop_duplicates("CustomerID").copy()
    products = products.drop_duplicates("ProductID").copy()
    orders = orders.drop_duplicates("OrderID").copy()
    order_details = order_details.drop_duplicates("DetailID").copy()
    reviews = reviews.drop_duplicates("ReviewID").copy()

    customers["City"] = customers["City"].str.strip().str.title()
    customers["RegistrationDate"] = pd.to_datetime(customers["RegistrationDate"], format="mixed")
    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(orders["OrderDate"], format="%d.%m.%Y", errors="coerce")
    )
    reviews["ReviewDate"] = pd.to_datetime(reviews["ReviewDate"], format="mixed")

    customers["Age"] = pd.to_numeric(customers["Age"], errors="coerce").fillna(customers["Age"].median())
    order_details["Quantity"] = pd.to_numeric(order_details["Quantity"], errors="coerce").fillna(order_details["Quantity"].median())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(order_details["Quantity"])
    axes[0].set_title("Boxplot Quantity (Преди)")

    z_outliers = (np.abs(zscore(order_details["Quantity"])) > 3).sum()
    q1, q3 = order_details["Quantity"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    print(f"Outliers -> Z-score: {z_outliers} | IQR [{lower_bound:.2f}, {upper_bound:.2f}]: {((order_details['Quantity'] < lower_bound) | (order_details['Quantity'] > upper_bound)).sum()}")

    order_details.loc[order_details["Quantity"] > upper_bound, "Quantity"] = max(1, int(round(upper_bound)))
    order_details["LineTotal"] = (order_details["Quantity"] * order_details["UnitPrice"]).round(2)

    axes[1].boxplot(order_details["Quantity"])
    axes[1].set_title("Boxplot Quantity (След)")
    plt.tight_layout()
    plt.show()

    after = {"Orders_rows": len(orders), "OrderDetails_missing": order_details["Quantity"].isna().sum(),
             "Max_Qty": order_details["Quantity"].max(), "Reviews_rows": len(reviews)}
    print("\nСравнение преди/след:\n", pd.DataFrame({"Преди": before, "След": after}))

    df_merged = (order_details.merge(orders, on="OrderID")
                 .merge(customers, on="CustomerID")
                 .merge(products.drop(columns=["UnitPrice"]), on="ProductID", suffixes=("_Customer", "_Product")))

    df_merged["Revenue"] = df_merged["LineTotal"]
    df_merged["TotalCost"] = (df_merged["Cost"] * df_merged["Quantity"]).round(2)
    df_merged["Profit"] = (df_merged["Revenue"] - df_merged["TotalCost"]).round(2)

    le = LabelEncoder()
    df_merged["City_Label"] = le.fit_transform(df_merged["City"])
    df_merged["CustomerCategory_Label"] = le.fit_transform(df_merged["Category_Customer"])
    df_merged = pd.concat([df_merged, pd.get_dummies(df_merged["PaymentMethod"], prefix="Pay", dtype=int)], axis=1)

    num_cols = ["Quantity", "UnitPrice", "Revenue", "Profit"]
    df_merged[[f"{c}_MinMax" for c in num_cols]] = MinMaxScaler().fit_transform(df_merged[num_cols])
    df_merged[[f"{c}_Std" for c in num_cols]] = StandardScaler().fit_transform(df_merged[num_cols])

    return customers, products, orders, order_details, reviews, df_merged


# етап 3: статистически анализ
def statistical_analysis(df_merged):
    print("\n--- ЕТАП 3: СТАТИСТИЧЕСКИ АНАЛИЗ ---")
    orders_val = df_merged.groupby("OrderID")["Revenue"].sum()
    metrics = {"UnitPrice": df_merged["UnitPrice"], "Quantity": df_merged["Quantity"], "OrderValue": orders_val}

    stats_df = pd.DataFrame([{
        "Показател": k, "Mean": round(s.mean(), 2), "Median": round(s.median(), 2), "Mode": round(s.mode().iloc[0], 2),
        "Var": round(s.var(), 2), "Std": round(s.std(), 2), "Min": round(s.min(), 2),
        "Q1": round(s.quantile(0.25), 2), "Q3": round(s.quantile(0.75), 2),
        "IQR": round(s.quantile(0.75) - s.quantile(0.25), 2), "Max": round(s.max(), 2)
    } for k, s in metrics.items()]).set_index("Показател")
    print(stats_df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, s) in zip(axes, metrics.items()):
        ax.hist(s, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(s.mean(), color="red", linestyle="--", label=f"Mean: {s.mean():.1f}")
        ax.axvline(s.median(), color="green", linestyle="-.", label=f"Median: {s.median():.1f}")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    mu, sigma = orders_val.mean(), orders_val.std()
    x_norm = np.linspace(max(0, mu - 3 * sigma), mu + 3 * sigma, 300)
    axes[0].hist(np.random.normal(mu, sigma, 1000), bins=25, density=True, alpha=0.6, color="skyblue", edgecolor="black")
    axes[0].plot(x_norm, norm.pdf(x_norm, mu, sigma), color="red", lw=2)
    axes[0].set_title(f"Нормално (P > mu+sigma: {1 - norm.cdf(mu + sigma, mu, sigma):.2f})")

    low_p, high_p = df_merged["UnitPrice"].min(), df_merged["UnitPrice"].max()
    axes[1].hist(np.random.uniform(low_p, high_p, 1000), bins=20, color="lightgreen", edgecolor="black")
    axes[1].set_title(f"Равномерно (P <= mid: {uniform.cdf((low_p + high_p) / 2, low_p, high_p - low_p):.2f})")

    p_comp = (df_merged.drop_duplicates("OrderID")["Status"] == "Completed").mean()
    x_bin = np.arange(0, 11)
    axes[2].bar(x_bin, binom.pmf(x_bin, 10, p_comp), color="salmon", edgecolor="black")
    axes[2].set_title(f"Биномиално (n=10, p={p_comp:.2f})")
    plt.tight_layout()
    plt.show()

    r_p, p_p = pearsonr(df_merged["UnitPrice"], df_merged["Profit"])
    r_s, _ = spearmanr(df_merged["UnitPrice"], df_merged["Profit"])
    print(f"\nUnitPrice vs Profit -> Pearson r={r_p:.4f} (p={p_p:.2e}) | Spearman rho={r_s:.4f}")

    corr_matrix = df_merged[["Age", "Quantity", "UnitPrice", "Cost", "Revenue", "Profit"]].corr()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title("Корелационна матрица")
    axes[1].scatter(df_merged["UnitPrice"], df_merged["Profit"], alpha=0.5, color="purple")
    axes[1].set_title("UnitPrice vs Profit")
    plt.tight_layout()
    plt.show()

    return stats_df, corr_matrix


# етап 4: времеви серии и прогнозиране
def time_series_analysis(df_merged):
    print("\n--- ЕТАП 4: ВРЕМЕВИ СЕРИИ И ПРОГНОЗИРАНЕ ---")
    valid_df = df_merged[df_merged["Status"] != "Cancelled"].copy()
    daily_ts = valid_df.groupby("OrderDate").agg(DailyRevenue=("Revenue", "sum"), OrderCount=("OrderID", "nunique")).sort_index()

    full_idx = pd.date_range("2025-11-01", "2026-10-31", freq="D")
    print(f"Период: 1 година ({len(full_idx)} дни) | Дни без поръчки: {len(full_idx) - len(daily_ts)}")
    daily_ts = daily_ts.reindex(full_idx, fill_value=0)
    weekly_ts = daily_ts.resample("W").sum()
    monthly_ts = daily_ts.resample("ME").sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(daily_ts.index, daily_ts["DailyRevenue"], alpha=0.4, label="Дневни")
    axes[0, 0].plot(daily_ts.index, daily_ts["DailyRevenue"].rolling(14).mean(), color="red", label="14-дн. MA")
    axes[0, 0].set_title("Дневни приходи и тренд")
    axes[0, 0].legend()

    axes[0, 1].plot(weekly_ts.index, weekly_ts["DailyRevenue"], marker="o", color="teal")
    axes[0, 1].set_title("Седмични приходи")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = daily_ts.groupby(daily_ts.index.day_name())["DailyRevenue"].mean().reindex(days)
    axes[1, 0].bar(dow.index, dow.values, color="coral", edgecolor="black")
    axes[1, 0].set_title("Сезонност по ден от седмицата")
    axes[1, 0].tick_params(axis="x", rotation=30)

    axes[1, 1].bar(monthly_ts.index.strftime("%Y-%m"), monthly_ts["DailyRevenue"], color="slateblue", edgecolor="black")
    axes[1, 1].set_title("Месечни приходи")
    axes[1, 1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    series = weekly_ts["DailyRevenue"]
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]

    naive_pred = np.repeat(train.iloc[-1], len(test))
    ma_pred = np.repeat(train.rolling(4).mean().iloc[-1], len(test))
    lr_pred = LinearRegression().fit(np.arange(len(train)).reshape(-1, 1), train.values).predict(
        np.arange(len(train), len(series)).reshape(-1, 1)
    )

    scaler = MinMaxScaler()
    tr_sc = scaler.fit_transform(train.values.reshape(-1, 1))
    te_sc = scaler.transform(test.values.reshape(-1, 1))
    win = 4
    X_tr = np.array([tr_sc[i:i + win] for i in range(len(tr_sc) - win)])
    Y_tr = np.array([tr_sc[i + win] for i in range(len(tr_sc) - win)])
    comb = np.concatenate([tr_sc[-win:], te_sc])
    X_te = np.array([comb[i:i + win] for i in range(len(comb) - win)])

    lstm = Sequential([tf.keras.Input(shape=(win, 1)), LSTM(32, activation="relu"), Dense(1)])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_tr, Y_tr, epochs=50, batch_size=8, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm.predict(X_te, verbose=0)).flatten()

    preds_map = {"Naive": naive_pred, "Moving Average": ma_pred, "Linear Regression": lr_pred, "LSTM": lstm_pred}
    eval_df = pd.DataFrame([{
        "Модел": m, "MAE": round(mean_absolute_error(test, p), 2),
        "MSE": round(mean_squared_error(test, p), 2), "RMSE": round(np.sqrt(mean_squared_error(test, p)), 2)
    } for m, p in preds_map.items()]).set_index("Модел")
    print("\nСравнение на моделите:\n", eval_df.to_string())

    plt.figure(figsize=(12, 4.5))
    plt.plot(train.index, train.values, label="Train", color="black", lw=2)
    plt.plot(test.index, test.values, label="Test", color="blue", marker="o", lw=2)
    for m, p in preds_map.items():
        plt.plot(test.index, p, label=m, linestyle="--")
    plt.title("Сравнение на прогнозни модели")
    plt.xlabel("Дата")
    plt.ylabel("Седмични приходи")
    plt.legend()
    plt.tight_layout()
    plt.show()

    lr_full = LinearRegression().fit(np.arange(len(series)).reshape(-1, 1), series.values)
    future_forecast = pd.DataFrame({
        "Date": pd.date_range(series.index[-1] + pd.Timedelta(days=7), periods=4, freq="W"),
        "PredictedRevenue": lr_full.predict(np.arange(len(series), len(series) + 4).reshape(-1, 1)).round(2)
    })
    return daily_ts, weekly_ts, monthly_ts, eval_df, future_forecast


# етап 5: анализ на клиентските мнения
def reviews_nlp_analysis(df_reviews):
    print("\n--- ЕТАП 5: АНАЛИЗ НА КЛИЕНТСКИТЕ МНЕНИЯ ---")
    reviews = df_reviews.copy()
    stop_words = set(ENGLISH_STOP_WORDS)

    reviews["CleanText"] = reviews["ReviewText"].apply(
        lambda t: " ".join([w for w in re.sub(r"http\S+|www\S+|@\w+|#|[^a-zA-Z\s]", " ", str(t).lower()).split()
                            if w not in stop_words and len(w) > 2])
    )

    cv = CountVectorizer().fit(reviews["CleanText"])
    word_freq = pd.DataFrame({"Word": cv.get_feature_names_out(), "Freq": cv.transform(reviews["CleanText"]).sum(axis=0).A1}).sort_values("Freq", ascending=False)

    bg = CountVectorizer(ngram_range=(2, 2)).fit(reviews["CleanText"])
    bigram_freq = pd.DataFrame({"Bigram": bg.get_feature_names_out(), "Freq": bg.transform(reviews["CleanText"]).sum(axis=0).A1}).sort_values("Freq", ascending=False)

    tfidf = TfidfVectorizer().fit(reviews["CleanText"])
    tfidf_df = pd.DataFrame({"Word": tfidf.get_feature_names_out(), "Score": tfidf.transform(reviews["CleanText"]).sum(axis=0).A1}).sort_values("Score", ascending=False)
    print("Топ 5 думи (TF-IDF):\n", tfidf_df.head(5).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].barh(word_freq.head(10)["Word"], word_freq.head(10)["Freq"], color="steelblue")
    axes[0].invert_yaxis()
    axes[0].set_title("Топ 10 думи")
    axes[1].barh(bigram_freq.head(10)["Bigram"], bigram_freq.head(10)["Freq"], color="darkorange")
    axes[1].invert_yaxis()
    axes[1].set_title("Топ 10 фрази (Bigrams)")
    plt.tight_layout()
    plt.show()

    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

    reviews["Sentiment"] = reviews["ReviewText"].apply(
        lambda t: "Positive" if sia.polarity_scores(str(t))["compound"] > 0.05
        else ("Negative" if sia.polarity_scores(str(t))["compound"] < -0.05 else "Neutral")
    )
    sentiment_summary = pd.DataFrame({
        "Брой мнения": reviews["Sentiment"].value_counts(),
        "Дял (%)": (reviews["Sentiment"].value_counts(normalize=True) * 100).round(2)
    })
    print("\nSentiment разпределение:\n", sentiment_summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"Positive": "seagreen", "Neutral": "gold", "Negative": "crimson"}
    axes[0].bar(sentiment_summary.index, sentiment_summary["Брой мнения"], color=[colors[c] for c in sentiment_summary.index])
    axes[0].set_title("Разпределение на Sentiment")

    monthly_sent = pd.crosstab(reviews["ReviewDate"].dt.to_period("M").astype(str), reviews["Sentiment"])
    monthly_sent.plot(kind="line", marker="o", ax=axes[1], color=[colors.get(c, "gray") for c in monthly_sent.columns])
    axes[1].set_title("Динамика на Sentiment по месеци")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    return reviews, sentiment_summary


# етап 6: клиентска сегментация  и бизнес анализ
def customer_segmentation_and_business_analysis(df_merged):
    print("\n--- ЕТАП 6: КЛИЕНТСКА СЕГМЕНТАЦИЯ И БИЗНЕС АНАЛИЗ ---")
    valid_df = df_merged[df_merged["Status"] != "Cancelled"].copy()
    snap = valid_df["OrderDate"].max() + pd.Timedelta(days=1)

    rfm = valid_df.groupby("CustomerID").agg(
        Recency=("OrderDate", lambda x: (snap - x.max()).days),
        Frequency=("OrderID", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalProfit=("Profit", "sum")
    ).reset_index()
    rfm["AOV"] = (rfm["Monetary"] / rfm["Frequency"]).round(2)

    rfm_scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    inertias = [KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(rfm_scaled).inertia_ for k in range(2, 9)]

    plt.figure(figsize=(7, 3.5))
    plt.plot(range(2, 9), inertias, marker="o", color="navy")
    plt.title("Elbow метод (K-means)")
    plt.xlabel("Брой клъстери (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()

    rfm["Cluster"] = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit_predict(rfm_scaled)
    profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary", "AOV", "TotalProfit"]].mean().round(2)
    profile["CustomersCount"] = rfm["Cluster"].value_counts()

    ranks = profile["Monetary"].rank(ascending=False)
    seg_map = {cid: ("Champions" if r == 1 else "Loyal" if r == 2 else "At Risk" if profile.loc[cid, "Recency"] > profile["Recency"].median() else "Promising")
               for cid, r in ranks.items()}
    rfm["Segment"] = rfm["Cluster"].map(seg_map)
    profile["Segment"] = profile.index.map(seg_map)
    print(profile[["Segment", "CustomersCount", "Recency", "Frequency", "Monetary", "AOV"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for seg, grp in rfm.groupby("Segment"):
        axes[0].scatter(grp["Frequency"], grp["Monetary"], label=seg, alpha=0.75)
    axes[0].set_title("Frequency vs Monetary")
    axes[0].legend()
    rfm["Segment"].value_counts().plot(kind="bar", ax=axes[1], color="teal", edgecolor="black", title="Клиенти по сегменти")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.show()

    cat_analysis = valid_df.groupby("Category_Product").agg(
        OrdersCount=("OrderID", "nunique"), UnitsSold=("Quantity", "sum"),
        Revenue=("Revenue", "sum"), Cost=("TotalCost", "sum"), Profit=("Profit", "sum")
    ).sort_values("Profit", ascending=False)
    cat_analysis["Margin_%"] = ((cat_analysis["Profit"] / cat_analysis["Revenue"]) * 100).round(2)
    print("\nАнализ по категории:\n", cat_analysis.round(2).to_string())

    df_merged = df_merged.merge(rfm[["CustomerID", "Segment"]], on="CustomerID", how="left")
    df_merged["Segment"] = df_merged["Segment"].fillna("No Orders")
    return rfm, profile, cat_analysis, df_merged


# етап 7: интерактивно табло
def build_interactive_dashboard(df_merged, rfm_df, reviews_analyzed, weekly_ts, future_forecast):
    print("\n--- ЕТАП 7: ИНТЕРАКТИВНО ТАБЛО  ---")
    valid_df = df_merged[df_merged["Status"] != "Cancelled"].copy()
    rev, prof = valid_df["Revenue"].sum(), valid_df["Profit"].sum()
    orders_cnt = valid_df["OrderID"].nunique()

    mid = valid_df["OrderDate"].min() + (valid_df["OrderDate"].max() - valid_df["OrderDate"].min()) / 2
    rev_h1 = valid_df[valid_df["OrderDate"] <= mid]["Revenue"].sum()
    rev_h2 = valid_df[valid_df["OrderDate"] > mid]["Revenue"].sum()

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}], [{"type": "xy", "colspan": 2}, None],
               [{"type": "domain"}, {"type": "domain"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("", "", "Седмични приходи и прогноза", "Клиентски сегменти", "Sentiment",
                        "Приходи и печалба по категории", "Месечен ръст (%)"),
        vertical_spacing=0.10
    )

    fig.add_trace(go.Indicator(mode="number+delta", value=rev_h2, number={"prefix": "$", "valueformat": ",.2f"},
                               delta={"reference": rev_h1, "relative": True},
                               title={"text": f"Приходи H2 vs H1 (Общо: ${rev:,.2f} | Марж: {prof / rev * 100:.1f}%)"}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=rev / orders_cnt, number={"prefix": "$", "valueformat": ",.2f"},
                               title={"text": f"AOV (Поръчки: {orders_cnt})"}), row=1, col=2)

    fig.add_trace(go.Scatter(x=weekly_ts.index, y=weekly_ts["DailyRevenue"], mode="lines+markers", name="Реални"), row=2, col=1)
    fig.add_trace(go.Scatter(x=future_forecast["Date"], y=future_forecast["PredictedRevenue"], mode="lines+markers", name="Прогноза", line=dict(dash="dash")), row=2, col=1)

    seg_c = rfm_df["Segment"].value_counts()
    fig.add_trace(go.Pie(labels=seg_c.index, values=seg_c.values, hole=0.4), row=3, col=1)

    sent_c = reviews_analyzed["Sentiment"].value_counts()
    fig.add_trace(go.Pie(labels=sent_c.index, values=sent_c.values, hole=0.4), row=3, col=2)

    cat_sum = valid_df.groupby("Category_Product")[["Revenue", "Profit"]].sum().sort_values("Revenue", ascending=False)
    fig.add_trace(go.Bar(x=cat_sum.index, y=cat_sum["Revenue"], name="Приходи"), row=4, col=1)
    fig.add_trace(go.Bar(x=cat_sum.index, y=cat_sum["Profit"], name="Печалба"), row=4, col=1)

    valid_df["Month"] = valid_df["OrderDate"].dt.to_period("M").astype(str)
    m_rev = valid_df.groupby("Month")["Revenue"].sum().reset_index()
    m_rev["Growth_%"] = (m_rev["Revenue"].pct_change() * 100).fillna(0)
    fig.add_trace(go.Bar(x=m_rev["Month"], y=m_rev["Growth_%"], name="Ръст %",
                         marker_color=np.where(m_rev["Growth_%"] >= 0, "#2ca02c", "#d62728")), row=4, col=2)

    cat_buttons = [dict(
        label="Всички категории",
        method="update",
        args=[{"x": [cat_sum.index, cat_sum.index], "y": [cat_sum["Revenue"], cat_sum["Profit"]]}, {}, [6, 7]]
    )]
    for cat in sorted(valid_df["Category_Product"].unique()):
        sub = valid_df[valid_df["Category_Product"] == cat]
        cat_buttons.append(dict(
            label=cat,
            method="update",
            args=[{"x": [[cat], [cat]], "y": [[sub["Revenue"].sum()], [sub["Profit"].sum()]]}, {}, [6, 7]]
        ))

    fig.update_layout(title_text="<b>ИНТЕРАКТИВНО БИЗНЕС ТАБЛО</b>", title_x=0.5, height=1300, barmode="group", template="plotly_white",
                      updatemenus=[dict(buttons=cat_buttons, direction="down", x=0.02, y=1.04)])
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    path = os.path.abspath("interactive_dashboard.html")
    fig.write_html(path)
    webbrowser.open(f"file://{path}")


# етап 8: изводи и бизнес препоръки чрез llm
def ask_llm_for_business_analysis(df_merged, rfm_df, category_analysis, sentiment_summary, future_forecast):
    print("\n" + "=" * 60)
    print("ЕТАП 8: БИЗНЕС АНАЛИЗ ОТ LLM")
    print("=" * 60)

    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY",
        ""#enter api key here
    )
    MODEL = "google/gemma-3-27b-it:free"

    valid_df = df_merged[df_merged["Status"] != "Cancelled"].copy()
    stats_dict = {
        "period": "1 year (2025-11-01 to 2026-10-31)",
        "total_orders": int(valid_df["OrderID"].nunique()),
        "total_customers": int(valid_df["CustomerID"].nunique()),
        "total_revenue": round(float(valid_df["Revenue"].sum()), 2),
        "total_profit": round(float(valid_df["Profit"].sum()), 2),
        "profit_margin_percent": round(float((valid_df["Profit"].sum() / valid_df["Revenue"].sum()) * 100), 2),
        "mean_aov": round(float(valid_df.groupby("OrderID")["Revenue"].sum().mean()), 2),
        "grouped_by_category": category_analysis[["Revenue", "Profit", "Margin_%"]].round(2).to_dict("index"),
        "grouped_by_city": valid_df.groupby("City").agg(
            Revenue_sum=("Revenue", "sum"), Profit_sum=("Profit", "sum"), Orders_count=("OrderID", "nunique")
        ).round(2).to_dict("index"),
        "customer_segments_rfm": rfm_df.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2).to_dict("index"),
        "reviews_sentiment_share_%": sentiment_summary["Дял (%)"].to_dict(),
        "next_4_weeks_forecast_sum": round(float(future_forecast["PredictedRevenue"].sum()), 2)
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ти си старши бизнес анализатор. Направи кратък, изпълнителен бизнес анализ на български език в 4 части: 1) Ключови наблюдения, 2) Силни страни, 3) Рискове/проблеми, 4) Конкретни препоръки. Базирай се САМО на подадените числа."},
            {"role": "user", "content": f"Статистики:\n{json.dumps(stats_dict, indent=2, ensure_ascii=False)}"}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    analysis_text = response.json()["choices"][0]["message"]["content"]
    print(analysis_text)
    return analysis_text


if __name__ == "__main__":
    generate_data()
    df_customers, df_products, df_orders, df_order_details, df_reviews = load_and_inspect_data()
    df_customers, df_products, df_orders, df_order_details, df_reviews, df_merged = preprocess_data(
        df_customers, df_products, df_orders, df_order_details, df_reviews
    )
    stats_df, corr_matrix = statistical_analysis(df_merged)
    daily_ts, weekly_ts, monthly_ts, eval_df, future_forecast = time_series_analysis(df_merged)
    df_reviews_analyzed, sentiment_summary = reviews_nlp_analysis(df_reviews)
    rfm_df, cluster_profile, category_analysis, df_merged = customer_segmentation_and_business_analysis(df_merged)
    build_interactive_dashboard(df_merged, rfm_df, df_reviews_analyzed, weekly_ts, future_forecast)
    ask_llm_for_business_analysis(df_merged, rfm_df, category_analysis, sentiment_summary, future_forecast)