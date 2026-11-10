# 🛒 What is Apriori? A Friendly Guide

> *"People who buy bread often buy butter too."* 🍞🧈
> Apriori is the algorithm that finds patterns like this, automatically, in thousands of shopping baskets.

---

## 🤔 The big idea in one minute

Imagine you run a small grocery store. Every day, customers walk out with a basket of items. After a month you have **thousands of receipts** 🧾.

You start wondering:

- Which products are often bought **together**?
- If someone puts **milk** in their basket, what should I suggest next?
- Should I place **chips** 🥔 next to **soda** 🥤 on the shelf?

**Apriori** answers these questions. It looks through every basket and finds groups of items that show up together **often enough to matter**. These patterns become **association rules** like:

```
{ milk, bread }  ➡️  { butter }
```

You can read this as *"customers who buy milk and bread are likely to buy butter too."*

This kind of analysis is often called **Market Basket Analysis** 🧺.

---

## 🧱 The building blocks

Let's use a tiny store with only **5 receipts**:

| Basket | Items |
|:---:|---|
| 1 | 🥛 milk, 🍞 bread |
| 2 | 🥛 milk, 🧈 butter |
| 3 | 🥛 milk, 🍞 bread, 🧈 butter |
| 4 | 🍞 bread, 🧈 butter |
| 5 | 🥛 milk, 🍞 bread, 🧈 butter, 🍫 chocolate |

A group of items, like `{milk, bread}`, is called an **itemset**.

### 1️⃣ Support: *"How common is it?"*

**Support** = the share of baskets that contain the itemset.

- `{milk}` appears in baskets 1, 2, 3, 5 → **4 / 5 = 80%**
- `{milk, bread}` appears in baskets 1, 3, 5 → **3 / 5 = 60%**
- `{chocolate}` appears only in basket 5 → **1 / 5 = 20%**

💡 *Low support means the pattern is rare, so it might just be a coincidence.*

### 2️⃣ Confidence: *"How reliable is the rule?"*

For a rule `A ➡️ B`, **confidence** = among baskets with A, how many also have B?

$$\text{confidence}(A \Rightarrow B) = \frac{\text{support}(A \cup B)}{\text{support}(A)}$$

Example: `{milk} ➡️ {bread}`
- Baskets with milk: 4
- Baskets with milk **and** bread: 3
- Confidence = **3 / 4 = 75%** ✅

### 3️⃣ Lift: *"Is it really a pattern, or just popularity?"*

Here's the catch ⚠️: if almost **everyone** buys bread, then *any* rule ending in bread will look confident, even if the first item has nothing to do with it.

**Lift** fixes this by comparing the rule with how popular B already is:

$$\text{lift}(A \Rightarrow B) = \frac{\text{confidence}(A \Rightarrow B)}{\text{support}(B)}$$

| Lift | Meaning |
|:---:|---|
| **> 1** 📈 | A makes B *more* likely. A real association! |
| **= 1** ➖ | A and B are independent. No relationship. |
| **< 1** 📉 | A makes B *less* likely. They tend *not* to go together. |

Example: `{milk} ➡️ {bread}` → 0.75 / 0.80 = **0.94**. Slightly below 1, so in this tiny store milk doesn't really push people towards bread. Confidence alone (75%) would have fooled us! 😮

---

## ⚙️ How the Apriori algorithm works

The hard part is that the number of possible item combinations **explodes** 💥. With just 169 products (like our dataset), there are over **800,000 possible pairs and triples**, and far more larger groups.

Apriori avoids checking them all with one clever rule, the **Apriori principle**:

> 🔑 **If an itemset is rare, every bigger itemset containing it is rare too.**

That makes sense: if `{caviar}` 🐟 appears in only 0.1% of baskets, then `{caviar, champagne}` 🍾 can't possibly appear more often. So we never need to check it.

### The step-by-step recipe 📝

```
Step 1 ➜ Count every single item. Keep the ones with support ≥ min_support.
Step 2 ➜ Combine the survivors into pairs. Count them. Keep the frequent ones.
Step 3 ➜ Combine the frequent pairs into triples. Count. Keep. ✂️ Prune the rest.
Step 4 ➜ Repeat until no new frequent itemsets appear.
Step 5 ➜ Turn the frequent itemsets into rules, and keep those with good confidence/lift.
```

Think of it like a **talent show** 🎤: only singers who pass the first round can form duets, and only strong duets can form trios. Nobody wastes time auditioning a trio whose members already failed solo.

---

## 🎛️ The knobs you can turn

| Setting | What it does | Too low 👇 | Too high 👆 |
|---|---|---|---|
| `min_support` | Minimum share of baskets an itemset must appear in | Thousands of rare, noisy patterns 🌪️ | Only obvious patterns, almost no pairs 😴 |
| `min_confidence` | Minimum reliability of a rule | Weak, unreliable rules | Very few rules |
| `min_lift` | Usually set to > 1 | Keeps meaningless rules | Keeps only surprising ones |

In [notebook 01](notebooks/01-apriori-grocery-recommender.ipynb) we used **`min_support = 1%`**, **confidence ≥ 20%** and **lift > 1**, which gave **231 rules** from 9,835 real grocery baskets.

---

## 🐍 Apriori in Python (5 lines!)

Using the [`mlxtend`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) library:

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

baskets = [["milk", "bread"], ["milk", "butter"], ["milk", "bread", "butter"],
           ["bread", "butter"], ["milk", "bread", "butter", "chocolate"]]

te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)

frequent = apriori(onehot, min_support=0.4, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.0)
```

---

## 🌍 Where is Apriori used?

- 🛒 **Retail:** product recommendations, shelf placement, bundle deals
- 🎬 **Streaming & e-commerce:** "frequently bought together" sections
- 🏥 **Healthcare:** finding symptoms or conditions that often occur together
- 🌐 **Web analytics:** pages that users tend to visit in the same session
- 🔐 **Security:** spotting combinations of events that often come before an incident

---

## ✅ Strengths and ⚠️ limitations

| 👍 Strengths | 👎 Limitations |
|---|---|
| Easy to understand and explain ("because you bought X") | Ignores **who** the customer is (no personalisation) |
| No training or labels needed | Ignores the **order** of purchases |
| Rules are transparent and easy to audit | Favours **popular** items; rare items rarely appear |
| Works well on small and medium datasets | Can be slow on huge catalogues (FP-Growth is a faster alternative) |

📊 **In our project:** ranked by confidence, the Apriori rules predicted a hidden basket item about as well as simply recommending best-sellers. Their real value was **explainability**. Curious how modern systems do better? See [notebook 02](notebooks/02-modern-recommenders-movielens.ipynb) 🚀.

---

## 🧠 Quick recap

- 🧺 **Itemset**: a group of items bought together
- 📊 **Support**: how often the itemset appears
- 🎯 **Confidence**: how often B follows A
- 📈 **Lift**: whether A *really* raises the chance of B (> 1 = yes!)
- ✂️ **Apriori principle**: rare small sets ➜ rare bigger sets, so skip them

---

## 📚 References

1. Agrawal, R., Imieliński, T. & Swami, A. (1993). *Mining association rules between sets of items in large databases.* ACM SIGMOD. (The paper that introduced association rule mining.)
2. Agrawal, R. & Srikant, R. (1994). *Fast algorithms for mining association rules.* VLDB. (The original Apriori algorithm.)
3. Han, J., Pei, J. & Yin, Y. (2000). *Mining frequent patterns without candidate generation.* ACM SIGMOD. (FP-Growth, a faster alternative.)
4. Tan, P.-N., Steinbach, M. & Kumar, V. *Introduction to Data Mining*, chapter on association analysis. (A beginner-friendly textbook explanation.)
5. Hahsler, M., Hornik, K. & Reutterer, T. (2006). *Implications of probabilistic data modeling for mining association rules.* Springer. (Source of the Groceries dataset used in this project.)
6. Raschka, S. [`mlxtend` documentation: apriori](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) and [association_rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/).

---

⬅️ [Back to the project README](README.md)
