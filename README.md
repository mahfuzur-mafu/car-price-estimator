# Car Price Range Estimation App

This project is designed to estimate a car price range suitable for customers based on their financial information. By inputting details such as age, salary, and net worth, the app calculates and advises a car price range tailored to the customer.

## Project Overview

The goal of this project is to provide a seamless interface for users to input their details and receive an estimated car price range based on their financial data. The app uses simple calculations and a clean user interface to deliver actionable insights.

## Key Steps

### 1. Data Preparation
- Loaded the car purchasing data from a CSV file (`car_purchasing.csv`).
  ```python
  import pandas as pd
  df = pd.read_csv("car_purchasing.csv", encoding="latin-1")
  ```
- Displayed basic dataset information including shape, column names, and data types using `info()` and `shape`.
- Inspected the first few rows using `head()`.
- Checked for and handled missing values (if any) using `isnull().sum()` and potentially `dropna()`.
- Checked for and removed duplicate entries (if any) using `drop_duplicates()`.

### 2. Exploratory Data Analysis (EDA)
- Calculated descriptive statistics for numerical columns using `describe()`.
- Analyzed the distribution of key features, including `age`, `annual salary`, `credit card debt`, and `net worth`, using histograms and scatter plots.
- Visualized relationships between features and the target variable (`car purchase amount`) using scatter plots.

- Calculated correlations between numerical features and the target variable using `corr()`.
- Explored potential categorical features like gender using `groupby()` and visualizations.

### 3. Data Visualization

Scatter plots: Visualized the relationships between numerical features (age, salary, net worth, credit and debt) and car purchase amount.

![image](https://github.com/user-attachments/assets/d2c71046-ca5c-457b-9534-fb15f2847fb1)

![image](https://github.com/user-attachments/assets/4ae6219a-01c6-4a2b-8c48-93b9d7a733ec)

![image](https://github.com/user-attachments/assets/e1dc71ba-7683-4d1d-a836-df163aec94cd)

![image](https://github.com/user-attachments/assets/aeeb7434-132a-4c05-90ab-b12515686cb0)




Histograms: Showed the distribution of age in the dataset.


![image](https://github.com/user-attachments/assets/19f1048d-6e99-4031-aa35-b30f981396ba)



Bar plots: Compared mean net worth and Mean annual salary by gender, and visualized mean/maximum car purchase amount by age range.

![image](https://github.com/user-attachments/assets/974fa483-73ad-4040-83ba-13d08d32d0e7)



### 3. Feature Engineering
- Created new features to potentially improve model performance:
  - Converted `gender` to a categorical variable (`n_gender`) using `apply()`.
  ```python
  df['n_gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
  ```
  - Grouped `age` into ranges (`age_range`) using a custom function and `apply()`.
  ```python
  def age_group(age):
      if age < 30:
          return 'Young'
      elif 30 <= age < 50:
          return 'Middle-aged'
      else:
          return 'Senior'
  df['age_range'] = df['age'].apply(age_group)
  ```
- Analyzed the impact of the new features on the target variable.

### 4. Model Building and Evaluation
- Split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.
  ```python
  from sklearn.model_selection import train_test_split
  X = df[['age', 'annual_salary', 'credit_card_debt', 'net_worth']]
  y = df['car_purchase_amount']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
- Scaled numerical features using `StandardScaler` from `sklearn.preprocessing`.
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  ```
- Trained three regression models:
  - Linear Regression
  ```python
  from sklearn.linear_model import LinearRegression
  lr = LinearRegression()
  lr.fit(X_train, y_train)
  ```
  - Support Vector Regression
  ```python
  from sklearn.svm import SVR
  svr = SVR()
  svr.fit(X_train, y_train)
  ```
  - Random Forest Regression
  ```python
  from sklearn.ensemble import RandomForestRegressor
  rf = RandomForestRegressor()
  rf.fit(X_train, y_train)
  ```
- Performed hyperparameter tuning for SVR and Random Forest Regression using `GridSearchCV` to optimize model performance.
- Evaluated model performance using metrics like **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** using functions from `sklearn.metrics`.

### 5. Model Selection and Saving
- Selected the best-performing model based on evaluation results. (In this case, it was likely Linear Regression).
- Saved the chosen model using `joblib.dump` for future use.
  ```python
  import joblib
  joblib.dump(lr, "model.pkl")
  joblib.dump(scaler, "scaler.pkl")
  ```

## Streamlit Application
- A Streamlit app was developed to allow users to input customer details and get a recommended car price range.
  ```python
  age =st.number_input('Age',min_value=18,max_value=90,value=40, step=1)
  annual_salary = st.number_input('Annual Salary', min_value=500, max_value=9999999999, value=12000, step=5000)
  net_worth = st.number_input('Net Worth', min_value=0, max_value=999999999, step=2000, value=100000)


  X = [age,annual_salary,net_worth]
  calculate =st.button('Calculate')
  st.divider()

  X_scaled = scaler.transform([X])

  if calculate:
    st.balloons()
    X_2 = np.array(X)
    X_array = scaler.transform([X_2])
    
    prediction = model.predict(X_array)
    
    # Check if prediction is less than 0
    if prediction[0] < 0:
        st.write("Prediction is: Negative")
    else:
        st.write(f"Prediction is: {prediction[0]:}")
    
    st.write("Advice: cars in the similar values")
       
  else:
    st.write("Enter values")
  ```

## Results

- The application provides quick and accurate car price range estimates based on user inputs.
- Screenshot of the app interface:

 <img width="708" alt="image" src="https://github.com/user-attachments/assets/f42e1efe-173e-484e-8e35-8e964c114bf7" />


## Future Work

- Improve the model by incorporating more complex features.
- Compare performance with advanced models like Gradient Boosting or Neural Networks.
- Deploy the app on a cloud platform for wider accessibility.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.



