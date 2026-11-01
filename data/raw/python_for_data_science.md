# Python for Data Science

## Core Libraries

NumPy provides fast, memory-efficient multi-dimensional array operations and is the numerical
foundation for most of the Python data science stack. Pandas builds on NumPy to provide the
DataFrame, a labeled 2D table structure with powerful indexing, filtering, grouping, and merging
operations for tabular data. Matplotlib and Seaborn are used for static data visualization, while
Plotly is commonly used for interactive charts.

## Pandas DataFrames

A DataFrame is Pandas' core data structure: a table with labeled rows (index) and columns, where
each column can hold a different data type. Common operations include `df.head()` to preview data,
`df.groupby()` to aggregate by category, `df.merge()` to join two DataFrames, and `df.dropna()` /
`df.fillna()` to handle missing values.

## Handling Missing Data

Missing data can be handled by removing rows/columns with missing values (`dropna`), filling them
with a statistic like the mean/median/mode (`fillna`), or using more advanced imputation methods
such as K-Nearest-Neighbors imputation or model-based imputation. The right approach depends on how
much data is missing and whether it is missing completely at random, missing at random, or missing
not at random.

## Virtual Environments

A virtual environment is an isolated Python environment with its own installed packages, separate
from the system-wide Python installation. Tools like `venv`, `conda`, and `poetry` are commonly
used to create and manage virtual environments so that different projects can use different,
non-conflicting package versions.

## List Comprehensions

A list comprehension is a concise way to create a list by applying an expression to each item in
an iterable, optionally with a filtering condition, e.g. `[x**2 for x in range(10) if x % 2 == 0]`
produces the squares of the even numbers from 0 to 9.

## Jupyter Notebooks

Jupyter Notebooks are an interactive coding environment that mixes executable code cells with
markdown text, visualizations, and output, making them popular for exploratory data analysis,
prototyping, and sharing reproducible data science work.

## scikit-learn

scikit-learn is a widely used Python library providing a consistent API (`fit`, `predict`,
`transform`) for classical machine learning algorithms, including classification, regression,
clustering, dimensionality reduction, model selection tools like `train_test_split` and
`GridSearchCV`, and preprocessing utilities like `StandardScaler` and `OneHotEncoder`.
