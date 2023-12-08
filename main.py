import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# data
df = pd.read_csv('anomaly.csv')


# Define the z-score calculation function
def calculate_z_score(user_input, feature_mean, feature_std):
    z_score = (user_input - feature_mean) / feature_std
    return z_score


# Define the function to check for fraud using Z-Score
def check_for_fraud(user_inputs):
    # Define group-wise features and their means and standard deviations
    customer_profile_features = ["Total_Relationship_Count", "Credit_Limit"]
    customer_engagement_features = ["Months_Inactive_12_mon", "Contacts_Count_12_mon"]
    credit_card_usage_features = ["Total_Revolving_Bal", "Avg_Open_To_Buy"]
    transaction_history_features = ["Total_Trans_Amt", "Total_Trans_Ct", "Avg_Utilization_Ratio"]

    #  mean and std values
    # Calculate the mean and standard deviation for a feature
    mean_1 = df['Total_Relationship_Count'].mean()
    std_1 = df['Total_Relationship_Count'].std()

    # Calculate the mean and standard deviation for a feature
    mean_2 = df['Credit_Limit'].mean()
    std_2 = df['Credit_Limit'].std()

    # Calculate the mean and standard deviation for a feature
    mean_3 = df['Months_Inactive_12_mon'].mean()
    std_3 = df['Months_Inactive_12_mon'].std()

    # Calculate the mean and standard deviation for a feature
    mean_4 = df['Contacts_Count_12_mon'].mean()
    std_4 = df['Contacts_Count_12_mon'].std()

    # Calculate the mean and standard deviation for a feature
    mean_5 = df['Total_Revolving_Bal'].mean()
    std_5 = df['Total_Revolving_Bal'].std()

    # Calculate the mean and standard deviation for a feature
    mean_6 = df['Avg_Open_To_Buy'].mean()
    std_6 = df['Avg_Open_To_Buy'].std()

    # Calculate the mean and standard deviation for a feature
    mean_7 = df['Total_Trans_Amt'].mean()
    std_7 = df['Total_Trans_Amt'].std()

    # Calculate the mean and standard deviation for a feature
    mean_8 = df['Total_Trans_Ct'].mean()
    std_8 = df['Total_Trans_Ct'].std()

    # Calculate the mean and standard deviation for a feature
    mean_9 = df['Avg_Utilization_Ratio'].mean()
    std_9 = df['Avg_Utilization_Ratio'].std()

    means = [mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, mean_8, mean_9]
    stds = [std_1, std_2, std_3, std_4, std_5, std_6, std_7, std_8, std_9]

    # Calculate z-scores for each group
    z_scores = [calculate_z_score(user_inputs[feature], mean, std) for feature, mean, std in zip(
        customer_profile_features + customer_engagement_features + credit_card_usage_features + transaction_history_features,
        means, stds
    )]

    # Check if more than one group has exceeded the z-score threshold
    exceeded_groups = 0
    if all(z >= 2 for z in z_scores[:2]):
        exceeded_groups += 1
        fraud_group = "Customer Profile"
    if all(z >= 2 for z in z_scores[2:4]):
        exceeded_groups += 1
        fraud_group = "Customer Engagement"
    if all(z >= 2 for z in z_scores[4:6]):
        exceeded_groups += 1
        fraud_group = "Credit Card Usage"
    if all(z >= 2 for z in z_scores[6:]):
        exceeded_groups += 1
        fraud_group = "Transaction History"

    # Determine the result based on the number of exceeded groups
    if exceeded_groups >= 2:
        return "Fraud in Multiple Groups"
    elif exceeded_groups == 1:
        return f"Fraud in {fraud_group}"
    else:
        return "Normal Transaction"


# Z-Score Page
def z_score():
    st.title("Z-Score Anomaly Detection")
    st.subheader("Enter values for Z-Score Anomaly Detection")

    # Collect user inputs for each feature
    user_inputs = {}
    for feature in ["Total_Relationship_Count", "Credit_Limit", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
                    "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Trans_Amt", "Total_Trans_Ct",
                    "Avg_Utilization_Ratio"]:
        user_input = st.number_input(f"Enter value for {feature}", step=1.0)
        user_inputs[feature] = user_input

        # Submit button
    if st.button("Submit"):
        # Call the check_for_fraud function with user inputs
        result = check_for_fraud(user_inputs)
        st.info("Result: {}".format(result))


# Sample Isolation Forest model (replace with your trained model)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df)


# Define the function to detect fraud using Isolation Forest
def detect_fraud(input_values):
    # Group the input values into categories
    customer_profile_features = ["Total_Relationship_Count", "Credit_Limit"]
    customer_engagement_features = ["Months_Inactive_12_mon", "Contacts_Count_12_mon"]
    credit_card_usage_features = ["Total_Revolving_Bal", "Avg_Open_To_Buy"]
    transaction_history_features = ["Total_Trans_Amt", "Total_Trans_Ct", "Avg_Utilization_Ratio"]

    # Create a DataFrame from user inputs
    user_data = pd.DataFrame({
        'Total_Relationship_Count': [input_values[0]],
        'Credit_Limit': [input_values[1]],
        'Months_Inactive_12_mon': [input_values[2]],
        'Contacts_Count_12_mon': [input_values[3]],
        'Total_Revolving_Bal': [input_values[4]],
        'Avg_Open_To_Buy': [input_values[5]],
        'Total_Trans_Amt': [input_values[6]],
        'Total_Trans_Ct': [input_values[7]],
        'Avg_Utilization_Ratio': [input_values[8]]
    })

    # Predict anomalies (-1) and inliers (1) for each category
    results = {}
    for category, features in zip(
            ["Customer Profile", "Customer Engagement", "Credit Card Usage", "Transaction History"],
            [customer_profile_features, customer_engagement_features, credit_card_usage_features,
             transaction_history_features]):
        # Fit the Isolation Forest model to the category
        model.fit(df[features])

        # Predict anomalies for the user data in the category
        category_predictions = model.predict(user_data[features])

        # Count the number of anomalies (-1)
        num_anomalies = sum(category_predictions == -1)

        # Store the result
        results[category] = num_anomalies

    return results


# Isolation Forest Page
def isolation_forest():
    st.title("Isolation Forest Anomaly Detection")
    st.subheader("Enter values for Isolation Forest Anomaly Detection")

    # Collect user inputs for each feature
    # Collect user inputs for each feature
    user_inputs = {}
    user_inputs[0] = st.number_input("Enter value for Total_Relationship_Count", step=1.0)
    user_inputs[1] = st.number_input("Enter value for Credit_Limit", step=1.0)
    user_inputs[2] = st.number_input("Enter value for Months_Inactive_12_mon", step=1.0)
    user_inputs[3] = st.number_input("Enter value for Contacts_Count_12_mon", step=1.0)
    user_inputs[4] = st.number_input("Enter value for Total_Revolving_Bal", step=1.0)
    user_inputs[5] = st.number_input("Enter value for Avg_Open_To_Buy", step=1.0)
    user_inputs[6] = st.number_input("Enter value for Total_Trans_Amt", step=1.0)
    user_inputs[7] = st.number_input("Enter value for Total_Trans_Ct", step=1.0)
    user_inputs[8] = st.number_input("Enter value for Avg_Utilization_Ratio", step=1.0)

    # Submit button
    if st.button("Submit"):
        # Detect fraud for each category using the user inputs
        fraud_results = detect_fraud(list(user_inputs.values()))

        # Determine which category has more anomalies
        most_fraudulent_category = max(fraud_results, key=fraud_results.get)
        st.info(
            f"The most fraudulent category is {most_fraudulent_category} with {fraud_results[most_fraudulent_category]} anomalies.")


cols = ['Total_Relationship_Count', 'Credit_Limit', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
        'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio']


# DBSCAN Anomaly Detection
def dbscan():
    st.title("DBSCAN Anomaly Detection")

    # User input for a new row
    new_row = {}
    for col in cols:
        new_row[col] = st.number_input(f"Enter the value for {col}:", value=0.0)

    # Add the user input as a new row to the dataset
    new_df = pd.DataFrame([new_row], columns=cols)

    # Append the new row to the original DataFrame
    global df
    df = pd.concat([df, new_df], ignore_index=True)

    # Button to trigger the DBSCAN analysis
    if st.button("Run DBSCAN"):
        # DBSCAN on each column
        outlier_cols = []
        for col in cols:
            db = DBSCAN(eps=0.5, min_samples=5).fit(df[[col]])  # Use the updated DataFrame
            if db.labels_[-1] == -1:
                outlier_cols.append(col)

        # Display result
        if outlier_cols:
            st.write(f"The last row is considered as an outlier in columns: {', '.join(outlier_cols)}")
        else:
            st.write("The last row is not considered as an outlier.")


# LOF Anomaly Detection
def lof():
    st.title("LOF Anomaly Detection")

    # User input for a new row
    new_row = {}
    for col in cols:
        new_row[col] = st.number_input(f"Enter the value for {col}:", value=0.0)

    # Add the user input as a new row to the dataset
    new_df = pd.DataFrame([new_row], columns=cols)

    # Append the new row to the original DataFrame
    global df
    df = pd.concat([df, new_df], ignore_index=True)

    # Button to trigger the LOF analysis
    if st.button("Run LOF"):
        # LOF on each column
        Lof = LocalOutlierFactor()
        scores = Lof.fit_predict(df[cols])

        # Check if the LOF score for the last row (newly appended row) is <= -3
        if scores[-1] <= -3:
            st.write("The last row is considered as an outlier.")
        else:
            st.write("The last row is not considered as an outlier.")


# Home Page Layout

# Define a default value for the page
# Define a default value for the page
default_page = "home"

# Initialize the session state
if "page" not in st.session_state:
    st.session_state.page = default_page


def home():
    st.title("Anomaly Detection App")
    st.subheader("Using Different Algorithms and Models")

    # Create a layout with 2 rows and 2 columns
    col1, col2 = st.columns(2)

    # Model 1 Explanation
    with col1:
        st.markdown("## Z-Score")

        # Create an expander for the explanation
        with st.expander("Click here for explanation"):
            st.write(
                "The code assesses the potential fraudulence of financial transactions based on specific transaction features. "
                "It gathers user inputs for various transaction features, including the total relationship count, credit limit, "
                "and months inactive in the last 12 months."
                "\n\n"
                "Subsequently, it calculates Z-scores for each feature to determine its deviation from the mean value. "
                "These features are then categorized into:"
                "\n\n"
                "1. **Customer Profile**"
                "\n2. **Customer Engagement**"
                "\n3. **Credit Card Usage**"
                "\n4. **Transaction History**"
                "\n\n"
                "The code evaluates if the Z-scores for any of these groups exceed a threshold of 2, suggesting potential fraud "
                "in that category."
                "\n\n"
                "If more than one group surpasses this threshold, it indicates **Fraud in Multiple Groups**. In the event that "
                "only one group exceeds the threshold, it specifies the category where potential fraud is detected, such as "
                "**Fraud in Customer Profile** or **Fraud in Credit Card Usage**."
                "\n\n"
                "If none of the Z-scores in any group exceed the threshold, it concludes that the transaction is **Normal**."
                "\n\n"
                "Finally, the code prints the final result to indicate whether the transaction is normal or potentially "
                "fraudulent, and if fraudulent, specifies the category it falls into."
            )

    # Model 2 Explanation
    with col2:
        st.markdown("## Isolation Forest")

        # Create an expander for the explanation
        with st.expander("Click here for explanation"):
            st.write(
                "The Isolation Forest method is used for anomaly detection based on user inputs for various transaction features."
                "\n\n"
                "The code first groups the input values into categories such as Customer Profile, Customer Engagement, "
                "Credit Card Usage, and Transaction History. These categories are defined by specific features:"
                "\n\n"
                "- **Customer Profile Features:** Total Relationship Count, Credit Limit"
                "\n- **Customer Engagement Features:** Months Inactive in the Last 12 Months, Contacts Count in the Last 12 Months"
                "\n- **Credit Card Usage Features:** Total Revolving Balance, Average Open to Buy"
                "\n- **Transaction History Features:** Total Transaction Amount, Total Transaction Count, Average Utilization Ratio"
                "\n\n"
                "Then, the user inputs are transformed into a DataFrame, and the Isolation Forest model is applied to each category. "
                "Anomalies (-1) and inliers (1) are predicted for each category based on the Isolation Forest model."
                "\n\n"
                "The code counts the number of anomalies (-1) in each category and stores the results. After the user clicks the "
                "'Submit' button, the code determines which category has the most anomalies, indicating the most fraudulent category. "
                "Finally, the result is displayed, indicating the most fraudulent category and the number of anomalies in that category."
            )

    # Model 3 Explanation
    with col1:
        st.markdown("## DB-SCAN ")

        # Create an expander for the explanation
        with st.expander("Click here for explanation"):
            st.write(
                "The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) method is utilized for anomaly detection "
                "based on user inputs for various features in financial transactions."
                "\n\n"
                "The code allows users to input values for specific features related to the transaction. These features include:"
                "\n\n"
                "- Total Relationship Count"
                "\n- Credit Limit"
                "\n- Months Inactive in the Last 12 Months"
                "\n- Contacts Count in the Last 12 Months"
                "\n- Total Revolving Balance"
                "\n- Average Open to Buy"
                "\n- Total Transaction Amount"
                "\n- Total Transaction Count"
                "\n- Average Utilization Ratio"
                "\n\n"
                "After the user enters values for these features and clicks the 'Run DBSCAN' button, the code adds the user input "
                "as a new row to the dataset."
                "\n\n"
                "The DBSCAN algorithm is then applied to each column independently. It identifies outliers in each column based on "
                "density and minimum samples parameters. If the last row (user input) is considered an outlier in any column, the column name is "
                "added to the list of outlier columns."
                "\n\n"
                "Finally, the code displays the result, indicating whether the last row (user input) is considered an outlier in any columns. "
                "If there are outlier columns, it specifies which columns they are; otherwise, it indicates that the last row(user inputs) is not "
                "considered an outlier."
            )

    # Model 4 Explanation
    with col2:
        st.markdown("## Local Outlier Factor")

        # Create an expander for the explanation
        with st.expander("Click here for explanation"):
            st.write(
                "The LOF (Local Outlier Factor) method is employed for anomaly detection based on user inputs for various "
                "features in financial transactions."
                "\n\n"
                "Users are prompted to input values for specific features related to the transaction. These features include:"
                "\n\n"
                "- Total Relationship Count"
                "\n- Credit Limit"
                "\n- Months Inactive in the Last 12 Months"
                "\n- Contacts Count in the Last 12 Months"
                "\n- Total Revolving Balance"
                "\n- Average Open to Buy"
                "\n- Total Transaction Amount"
                "\n- Total Transaction Count"
                "\n- Average Utilization Ratio"
                "\n\n"
                "After the user enters values for these features and clicks the 'Run LOF' button, the code adds the user input "
                "as a new row to the dataset."
                "\n\n"
                "The LOF algorithm is then applied to the entire dataset, including the newly appended row. It calculates LOF "
                "scores, which measure the local density deviation of a data point with respect to its neighbors. The code checks "
                "if the LOF score for the last row (newly appended row) is less than or equal to -3. A score below this threshold "
                "indicates that the last row is considered an outlier."
                "\n\n"
                "Finally, the code displays the result, indicating whether the last row is considered an outlier based on the "
                "LOF score. If it is, the code specifies that the last row is considered an outlier; otherwise, it indicates that "
                "the last row is not considered an outlier."
            )

    # Add more models as needed


# Sidebar Menu
st.sidebar.title("Menu")

# Sidebar buttons for Home, Z-Score, and Isolation Forest
if st.sidebar.button("Home"):
    st.session_state.page = "home"

if st.sidebar.button("Z-Score"):
    st.session_state.page = "z_score"

if st.sidebar.button("Isolation Forest"):
    st.session_state.page = "isolation_forest"

if st.sidebar.button("DBSCAN"):
    st.session_state.page = "dbscan"

if st.sidebar.button("LOF"):
    st.session_state.page = "lof"

# Main App Logic
if st.session_state.page == "home":
    home()

elif st.session_state.page == "z_score":
    z_score()  # You need to define the z_score() function with its content

elif st.session_state.page == "isolation_forest":
    isolation_forest()

elif st.session_state.page == "dbscan":
    dbscan()

elif st.session_state.page == "lof":
    lof()
