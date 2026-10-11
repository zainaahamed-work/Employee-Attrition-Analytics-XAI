# Streamlit app: Employee Attrition Analytics Using Explainable AI
# Runs SHAP, K-Means clustering and Apriori rules on an uploaded HR CSV

# Libraries used by the app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Page title in the browser tab and a wide layout
st.set_page_config(page_title='Employee Attrition Analytics', layout='wide')


# ---------- Data preparation ----------
def prepare_data(raw):
    # Work on a copy so the uploaded data is never changed
    df = raw.copy()
    # Remove columns that carry no information or are only IDs
    df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], errors='ignore')
    # Convert Attrition from Yes/No into 1/0
    y = (df['Attrition'] == 'Yes').astype(int)
    # Keep every other column as input features
    X = df.drop(columns=['Attrition'])
    # Convert text columns into numbers using label encoding
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    return df, X, y


# ---------- Technique 1: SHAP ----------
@st.cache_data(show_spinner=False)
def compute_shap(raw):
    df, X, y = prepare_data(raw)
    # A small Random Forest is trained only so SHAP has something to explain
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    # Calculate SHAP values for every employee
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    # Keep only the values that explain the "left the company" class
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
    # Average absolute SHAP value per feature gives the importance ranking
    importance = pd.DataFrame({'Feature': X.columns,
                               'SHAP_Importance': np.abs(values).mean(axis=0)})
    importance = importance.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
    return X, values, importance


def show_shap_plots(X, values):
    # Bar chart of overall feature importance
    st.subheader('SHAP summary plot: which factors matter most')
    plt.figure()
    shap.summary_plot(values, X, plot_type='bar', show=False)
    st.pyplot(plt.gcf())
    plt.close()
    # Beeswarm plot showing the direction of each factor's influence
    st.subheader('SHAP beeswarm plot: direction of influence')
    plt.figure()
    shap.summary_plot(values, X, show=False)
    st.pyplot(plt.gcf())
    plt.close()


# ---------- Technique 2: K-Means clustering ----------
@st.cache_data(show_spinner=False)
def compute_clusters(raw, k):
    df, X, y = prepare_data(raw)
    # Scale all features to the 0-1 range so no feature dominates
    scaled = MinMaxScaler().fit_transform(X)
    # Group employees into k clusters
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    out = df.copy()
    out['Cluster'] = labels
    out['AttritionFlag'] = y.values
    out['OverTimeFlag'] = (out['OverTime'] == 'Yes').astype(int)
    # Summarise each cluster: size, attrition rate and average characteristics
    profile = out.groupby('Cluster').agg(
        Employees=('Cluster', 'size'),
        Attrition_Rate_Pct=('AttritionFlag', lambda s: s.mean() * 100),
        Avg_Age=('Age', 'mean'),
        Avg_MonthlyIncome=('MonthlyIncome', 'mean'),
        Avg_YearsAtCompany=('YearsAtCompany', 'mean'),
        Avg_JobLevel=('JobLevel', 'mean'),
        Overtime_Pct=('OverTimeFlag', lambda s: s.mean() * 100)
    ).round(1).reset_index()
    return profile


# ---------- Technique 3: Apriori association rules ----------
@st.cache_data(show_spinner=False)
def compute_rules(raw):
    df, X, y = prepare_data(raw)
    # Turn continuous columns into readable categories
    items = pd.DataFrame()
    items['Income'] = pd.qcut(df['MonthlyIncome'], 3, labels=['Low income', 'Medium income', 'High income'])
    items['Age'] = pd.cut(df['Age'], bins=[0, 29, 45, 100], labels=['Young (under 30)', 'Mid-career', 'Senior'])
    items['Tenure'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 10, 100],
                             labels=['New (2 yrs or less)', 'Experienced', 'Veteran'])
    items['OverTime'] = np.where(df['OverTime'] == 'Yes', 'Works overtime', 'No overtime')
    items['Marital'] = df['MaritalStatus']
    items['WorkLife'] = df['WorkLifeBalance'].map({1: 'Poor work-life balance', 2: 'Fair work-life balance',
                                                   3: 'Good work-life balance', 4: 'Excellent work-life balance'})
    # One True/False column per category, the format Apriori needs
    basket = pd.get_dummies(items.astype(str), prefix='', prefix_sep='').astype(bool)
    basket['Attrition'] = (y == 1).values
    # Find combinations that appear in at least 5% of employees
    frequent = apriori(basket, min_support=0.05, use_colnames=True, max_len=3)
    # Turn them into rules; newer mlxtend versions need the number of itemsets
    try:
        rules = association_rules(frequent, metric='confidence', min_threshold=0.0, num_itemsets=len(basket))
    except TypeError:
        rules = association_rules(frequent, metric='confidence', min_threshold=0.0)
    # Keep only rules whose result is Attrition
    rules = rules[rules['consequents'].apply(lambda c: c == frozenset({'Attrition'}))]
    # Highest confidence first, keep the top 10
    rules = rules.sort_values('confidence', ascending=False).head(10).reset_index(drop=True)
    table = pd.DataFrame({
        'Rank': range(1, len(rules) + 1),
        'Rule': rules['antecedents'].apply(lambda s: ' + '.join(sorted(s))),
        'Support_Pct': (rules['support'] * 100).round(1),
        'Confidence_Pct': (rules['confidence'] * 100).round(1),
        'Lift': rules['lift'].round(2)
    })
    return table


# ---------- HR report (rule-based, no paid API) ----------
actions = {
    'OverTime': 'Review overtime workload and staffing levels. Overtime is the strongest attrition driver in this analysis.',
    'StockOptionLevel': 'Extend stock option or long-term incentive schemes to more employees, as low stock option levels are linked with leaving.',
    'MonthlyIncome': 'Check pay against market rates, especially for lower-paid and early-career roles.',
    'Age': 'Offer career-growth and mentoring programmes aimed at younger employees, who leave more often.',
    'TotalWorkingYears': 'Create clearer progression paths for employees with limited total experience.',
    'JobLevel': 'Review promotion criteria and timelines so junior employees can see how to move up.',
    'DistanceFromHome': 'Consider flexible hours, remote days or commute support for employees who live far away.',
    'EnvironmentSatisfaction': 'Run regular pulse surveys on the work environment and act on the feedback.',
    'NumCompaniesWorked': 'Pay extra attention to onboarding and engagement for employees who have changed employers often.',
    'YearsAtCompany': 'Hold stay-interviews at key tenure points to understand what keeps people.'
}


def generate_hr_report(shap_table, rules_table, top_n_factors=5, top_n_rules=3):
    # Top factors from SHAP and top rules from Apriori
    top_factors = shap_table.sort_values('SHAP_Importance', ascending=False).head(top_n_factors)
    top_rules = rules_table.sort_values('Rank').head(top_n_rules)
    lines = ['HR ATTRITION RECOMMENDATIONS', '=' * 30, '']
    lines.append('1. Key factors linked to attrition (from SHAP)')
    for position, row in enumerate(top_factors.itertuples(), start=1):
        action = actions.get(row.Feature, 'Review this factor with HR leadership.')
        lines.append(f'   {position}. {row.Feature} (importance {row.SHAP_Importance:.3f}): {action}')
    lines.append('')
    lines.append('2. Combinations to watch (from Apriori rules)')
    for row in top_rules.itertuples():
        lines.append(f'   - {row.Rule}: {row.Confidence_Pct:.1f}% of these employees left (lift {row.Lift:.2f}).')
    lines.append('')
    lines.append('3. Note')
    lines.append('   These findings show patterns in past data, not guaranteed causes. Use them to guide conversations and further review.')
    lines.append('   Report generated from the SHAP and Apriori outputs using a rule-based template.')
    return '\n'.join(lines)


# ---------- Page layout ----------
st.title('Employee Attrition Analytics Using Explainable AI')
st.write('Upload an HR dataset (CSV) to see which factors drive attrition, which employee groups are at risk, and which combinations of traits lead to leaving.')

uploaded = st.file_uploader('Upload HR CSV', type='csv')

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.success(f'Loaded {raw.shape[0]} employees and {raw.shape[1]} columns.')

    tab1, tab2, tab3, tab4 = st.tabs(['SHAP', 'Clusters', 'Association rules', 'HR report'])

    with tab1:
        X, values, importance = compute_shap(raw)
        show_shap_plots(X, values)
        st.dataframe(importance)

    with tab2:
        k = st.slider('Number of clusters', 2, 8, 4)
        st.dataframe(compute_clusters(raw, k))

    with tab3:
        rules_table = compute_rules(raw)
        st.dataframe(rules_table)

    with tab4:
        _, _, importance = compute_shap(raw)
        rules_table = compute_rules(raw)
        st.text(generate_hr_report(importance, rules_table))
else:
    st.info('Please upload a CSV file to begin.')
