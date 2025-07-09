# importing the basic libaries for our analysis
import warnings  
warnings.filterwarnings("ignore")  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing the dataset we will be working on with
data = pd.read_csv('/kaggle/input/sleep-health-and-lifestyle-dataset/Sleep_health_and_lifestyle_dataset.csv')
print(data.shape)
data.head()
#to get some basic statistics 
data.describe()
#to get some basic statistics of the object datatype data
data.describe(include='object')
#checking the unique values in BMI Category column
data['BMI Category'].unique()
#checking the unique values in Occupation column
data['Occupation'].unique()
#to get overall structure of the dataset with there data types and total number of values
data.info()
#dropping any duplicate values if exists
data.drop_duplicates()
print(data.shape)
#to get the number of missing values
data.isnull().sum()
#replacing the NaN values to "None"
data['Sleep Disorder'].fillna('None', inplace=True)
print(data['Sleep Disorder'].unique())
# replacing the "Normal Weight" category to "Underweight"
data['BMI Category']= data['BMI Category'].replace('Normal Weight','Underweight')
print(data['BMI Category'].unique())
gender_counts = data['Gender'].value_counts()

# Plotting the donut chart (gender distribution)
plt.figure(figsize=(4, 4))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.8)

# Draw a white circle in the center to create the donut hole
centre_circle = plt.Circle((0, 0), 0.60, fc='white')  
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Gender Distribution')
#Plotting a histogram plot (Age by Gender)
sns.histplot(data=data,
           x='Age',
            hue='Gender',
            multiple='stack',
            kde=True)
plt.title('Gender Distribution by Age')
plt.figure(figsize=(8,4))
fig,axes=plt.subplots(1,2,figsize=(15,4))

#plotting bar graph (count of sleep disorder by gender)
sns.countplot(data=data,
            x= "Sleep Disorder",
            hue= 'Gender',
            ax=axes[0])

axes[0].set_title('Count of Sleep Disorder by Gender')
axes[0].set_xlabel('Sleep Disorder')
axes[0].set_ylabel('Count')

axes[0].grid(color='grey', linestyle='--',axis='y')

#plotting boxplot (stress level distribution by gender)
sns.boxplot(data=data,
           x='Gender',
           y='Stress Level',
           ax=axes[1])

axes[1].set_title('Stress Level Distribution by Gender')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Stress Level')

axes[1].grid(color='grey', linestyle='--',axis='y')
plt.figure(figsize=(10,4))

#plotting the bargraph (sleep duration vs physical activity)
sns.barplot(data=data,
           x='Sleep Duration',
           y='Physical Activity Level',
           errorbar=('ci',False))

plt.ylabel('Physical Activity(Minutes)')
plt.xlabel('Sleep Duration(Hours)')
plt.title('Sleep Duration by Physical Activity')
plt.xticks(rotation=45, horizontalalignment='right')

plt.grid(color='grey', linestyle='--',axis='y')
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

#plotting the histogram graph (age vs sleep disorder)
sns.histplot(data=data, 
             x='Age', 
             hue='Sleep Disorder', 
             multiple='stack', 
            kde=True ,ax=axes[0])

axes[0].set_title('Age Distribution by Sleep Disorders')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

axes[0].grid(color='grey', linestyle='--',axis='y')

#plotting the violin graph (age vs sleep disorder)
sns.violinplot(data=data,
               x='Age', 
               y='Sleep Disorder', 
               ax=axes[1])

axes[1].set_title('Violin of Age by Sleep Disorders')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Sleep Disorder')

axes[1].grid(color='grey', linestyle='--',axis='x')
#plotting violin graph (physical activity level vs sleep disorder)
sns.violinplot(data=data,
              x='Physical Activity Level',
              y='Sleep Disorder')

plt.title('Physical Activity Level by Sleep Disorders')
plt.xlabel('Physical Activity Level')
plt.ylabel('Sleep Disorder')

plt.grid(color='grey', linestyle='--',axis='x')
cross_tab = pd.crosstab(data['Occupation'], data['Sleep Disorder'])

# plotting the heatmap (sleep Disorder vs ocupation)
plt.figure(figsize=(5, 4))
sns.heatmap(cross_tab, annot=True, cmap='plasma')

plt.title('Sleep Disorder vs. Occupation Heatmap')
plt.xlabel('Sleep Disorder')
plt.ylabel('Occupation')
cross_tab = pd.crosstab(data['Occupation'], data['Physical Activity Level'])

# plotting the heatmap (sleep Disorder vs ocupation)
plt.figure(figsize=(5, 4))
sns.heatmap(cross_tab, annot=True, cmap='plasma')

plt.title('Physical Activity Level vs. Occupation Heatmap')
plt.xlabel('Physical Activity Level')
plt.ylabel('Occupation')
plt.figure(figsize=(8,4))

#plotting bar graph (Quality of sleep vs stress level)
sns.scatterplot(data=data,
           x='Quality of Sleep',
           y='Stress Level')

plt.title('Quality of Sleep by Stress Level')
plt.xlabel('Quality of Sleep')
plt.ylabel('Stress Level')

plt.grid(color='grey', linestyle='--')
category= data['BMI Category'].value_counts()

#plotting pie graph (BMI Category distribution)
plt.pie(category, labels= category.index, autopct='%1.1f%%', pctdistance=0.8)

plt.title('BMI Category Distribution')
fig,axes=plt.subplots(1,2, figsize=(15,4))

#plotting count of sleep disorder by BMI Category
sns.countplot(data=data,
           x='BMI Category',
           hue='Sleep Disorder',
             ax=axes[0])

axes[0].set_title('Count of Sleep Disorder by BMI Category')
axes[0].set_xlabel('BMI Category')
axes[0].set_ylabel('Count')

axes[0].grid(color='grey', linestyle='--',axis='y')

#plotting Box graph (BMI category vs stress level)
sns.boxplot(data=data,
           x='BMI Category',
           y='Stress Level',
           ax=axes[1])

axes[1].set_title('Stress Level by BMI Category')
axes[1].set_xlabel('BMI Category')
axes[1].set_ylabel('Stress Level')

axes[1].grid(color='grey', linestyle='--',axis='y')
#plotting the histogram graph (heart rate distribution by sleep disorder)
sns.histplot(data=data,
            x='Heart Rate',
             hue='Sleep Disorder',
             multiple='stack',
            kde=True,bins=10
            )

plt.title('Heart Rate Distribution by Sleep Disorder')
plt.xlabel('Heart Rate')
plt.ylabel('Count')

plt.grid(color='grey', linestyle='--',axis='y')
#plotting histogram graph (Hearrate distribution by gender)
sns.histplot(data=data,
            x='Heart Rate',
            hue='Gender',
            multiple='stack',
            kde=True,
            bins=10)

plt.title('Heart Rate Distribution by Gender')
plt.xlabel('Heart Rate')
plt.ylabel('Count')

plt.grid(color='grey', linestyle='--',axis='y')
#splitting Blood pressure column to Systolic and Diastolic columns
data[['Systolic','Diastolic']]=data['Blood Pressure'].str.split('/',expand=True).astype(int)

#plotting scatterplot (systolic vs diastolic)
sns.scatterplot(data=data,
               x='Systolic',
               y='Diastolic')

plt.grid( color='gray', linestyle='--', linewidth=0.7)
fig,axes=plt.subplots(1,2,figsize=(15,4))

#plotting box plot (systolic vs gender)
sns.boxplot(data=data,
           y='Systolic',
           x='Gender',
           ax=axes[0])

axes[0].set_title('Systolic Blood Pressure by Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Systolic Blood Pressure(mmHg)')

axes[0].grid(color='grey', linestyle='--',axis='y')

#plotting box plot (diastolic vs gender)
sns.boxplot(data=data,
           y="Diastolic",
           x='Gender',
           ax=axes[1])

axes[1].set_title('Diastolic Blood Pressure by Gender')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Diastolic Blood Pressure (mmHg)')

axes[1].grid(color='grey', linestyle='--',axis='y')
fig,axes=plt.subplots(1,2,figsize=(15,4))

#plotting box plot (systolic vs sleep disorder)
sns.boxplot(data=data,
           y='Systolic',
           x='Sleep Disorder',
           ax=axes[0])

axes[0].set_title('Systolic Blood Pressure by Sleep Disorder')
axes[0].set_title('Sleep Disorder')
axes[0].set_ylabel('Systolic Blood Pressure(mmHg)')

axes[0].grid(color='grey', linestyle='--',axis='y')

#plotting box plot (diastolic vs sleep disorder)
sns.boxplot(data=data,
           y="Diastolic",
           x='Sleep Disorder',
           ax=axes[1])

axes[1].set_title('Diastolic Blood Pressure by Sleep Disorder')
axes[1].set_xlabel('Sleep Disorder')
axes[1].set_ylabel('Diastolic Blood Pressure (mmHg)')

axes[1].grid(color='grey', linestyle='--',axis='y')
fig,axes=plt.subplots(1,2,figsize=(15,4))

#plotting box plot (systolic vs BMI category)
sns.boxplot(data=data,
           y='Systolic',
           x='BMI Category',
           ax=axes[0])

axes[0].set_title('Systolic Blood Pressure by BMI Category')
axes[0].set_xlabel('BMI Category')
axes[0].set_ylabel('Systolic Blood Pressure(mmHg)')

axes[0].grid(color='grey', linestyle='--')

#plotting box plot (diastolic vs BMI category)
sns.boxplot(data=data,
           y="Diastolic",
           x='BMI Category',
           ax=axes[1])

axes[1].set_title('Diastolic Blood Pressure by BMI Category')
axes[1].set_xlabel('BMI Category')
axes[1].set_ylabel('Diastolic Blood Pressure (mmHg)')

axes[1].grid(color='grey', linestyle='--')


plt.xlabel('Systolic Blood pressure (mmHg)')
plt.ylabel('Diastolic Blood pressure (mmHg)')
plt.title("Systolic vs Diastolic Scatter Plot")

