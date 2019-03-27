import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#import os
df=pd.read_csv('master.csv')
print(df)
#show data first 5 rows
df.head()
df.tail()
#random rows in dataset
df.sample(5)
df.describe()
df.iloc[:,1:5].describe()
#The info function shows the data types and numerical values of the features in our data set.
df.info()
#We will now set the headings of the feature values in the data set.
df.columns
df=df.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
df.columns
#And, how many rows and columns are there for all data?
df.shape
df.isnull().any()
df.isnull().values.any()
#Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.
df.isnull().sum()
#As you can see, most of the HDIForYear value is empty. That's why I want this value deleted.
df=df.drop(['HDIForYear','CountryYear'],axis=1)
#Now start analysis, min year and max year will find them
min_year=min(df.Year)
max_year=max(df.Year)
print('Min Year :',min_year)
print('Max Year :',max_year)
data_country=df[(df['Year']==min_year)]
country_1985=df[(df['Year']==min_year)].Country.unique()
country_1985_male=[]
country_1985_female=[]
for country in country_1985:
    country_1985_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_1985_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')])) 
plt.figure(figsize=(10,10))
sns.barplot(y=country_1985,x=country_1985_male,color='red')
sns.barplot(y=country_1985,x=country_1985_female,color='yellow')
plt.ylabel('Countries')
plt.xlabel('Count Male vs Female')
plt.title('1985 Year Suicide Rate Gender')
plt.show()

#Very odd all the rates came on an equal level. So let's do max year.

data_country=df[(df['Year']==max_year)]

country_2016=df[(df['Year']==max_year)].Country.unique()
country_2016_male=[]
country_2016_female=[]

for country in country_2016:
    country_2016_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_2016_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')])) 
    
#We found the ratio of men and women who committed suicide in some countries in 1985 and we are now charting.

plt.figure(figsize=(10,10))
sns.barplot(y=country_2016,x=country_2016_male,color='red')
sns.barplot(y=country_2016,x=country_2016_female,color='yellow')
plt.ylabel('Countries')
plt.xlabel('Count Male vs Female')
plt.title('2016 Year Suicide Rate Gender')
plt.show()
data_country=df[(df['Year']==min_year)]

country_1985_population=[]

for country in country_1985:
    country_1985_population.append(sum(data_country[(data_country['Country']==country)].Population))    

#Now year 1985 find sum population every country

plt.figure(figsize=(10,10))
sns.barplot(y=country_1985,x=country_1985_population)
plt.xlabel('Population Count')
plt.ylabel('Countries')
plt.title('1985 Year Sum Population for Suicide Rate')
plt.show()

#######################################################

data_country=df[(df['Year']==max_year)]

country_2016_population=[]

for country in country_2016:
    country_2016_population.append(sum(data_country[(data_country['Country']==country)].Population))    

#Now year 1985 find sum population every country

plt.figure(figsize=(10,10))
sns.barplot(y=country_2016,x=country_2016_population)
plt.xlabel('Population Count')
plt.ylabel('Countries')
plt.title('2016 Year Sum Population for Suicide Rate')
plt.show()
data_country=df[(df['Year']==min_year)]

data_age_5_14=[]
data_age_15_24=[]
data_age_25_34=[]


data_age_35_54=[]
data_age_55_74=[]
data_age_75=[]

for country in country_1985:
        data_age_5_14.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='5-14 years')]))
        data_age_15_24.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='15-24 years')]))
        data_age_25_34.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='25-34 years')]))
        data_age_35_54.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='35-54 years')]))
        data_age_55_74.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='55-74 years')]))
        data_age_75.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='75+ years')]))
        

#######################################################

data_country=df[(df['Year']==max_year)]

data_age_5_14=[]
data_age_15_24=[]
data_age_25_34=[]
data_age_35_54=[]
data_age_55_74=[]
data_age_75=[]

for country in country_2016:
        data_age_5_14.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='5-14 years')]))
        data_age_15_24.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='15-24 years')]))
        data_age_25_34.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='25-34 years')]))
        data_age_35_54.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='35-54 years')]))
        data_age_55_74.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='55-74 years')]))
        data_age_75.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='75+ years')]))
        
#there is an equal rate. We need to make the query process a little more complicated.

sns.countplot(df.Gender)
plt.show()
df.groupby('Age')['Gender'].count()
sns.barplot(x=df.groupby('Age')['Gender'].count().index,y=df.groupby('Age')['Gender'].count().values)
plt.show()
suicidesNo=[]
for country in df.Country.unique():
    suicidesNo.append(sum(df[df['Country']==country].SuicidesNo))   

suicidesNo=pd.DataFrame(suicidesNo,columns=['suicidesNo'])
country=pd.DataFrame(df.Country.unique(),columns=['country'])
data_suicide_countr=pd.concat([suicidesNo,country],axis=1)
#sns.barplot(x=data.Country.unique(),y=suicidesNo) 
#plt.show()

data_suicide_countr=data_suicide_countr.sort_values(by='suicidesNo',ascending=False)

sns.barplot(y=data_suicide_countr.country[:15],x=data_suicide_countr.suicidesNo[:15])
plt.show()
grouop_data=df.groupby(['Age','Gender'])['SuicidesNo'].sum().unstack()
grouop_data=grouop_data.reset_index().melt(id_vars='Age')

grouop_data_female=grouop_data.iloc[:6,:]
grouop_data_male=grouop_data.iloc[6:,:]

grouop_data_female
grouop_data_male
female_=[175437,208823,506233,16997,430036,221984]
male_=[633105,915089,1945908,35267,1228407,431134]
plot_id = 0
for i,age in enumerate(['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years']):
    plot_id += 1
    plt.subplot(3,2,plot_id)
    plt.title(age)
    fig, ax = plt.gcf(), plt.gca()
    sns.barplot(x=['female','male'],y=[female_[i],male_[i]],color='blue')
    plt.tight_layout()
    fig.set_size_inches(10, 15)
plt.show() 
sns.countplot(df.Generation)
plt.title('Generation Counter')
plt.xticks(rotation=45)
plt.show()
df.head()

sns.set(style='whitegrid')
sns.boxplot(df['Population'])
plt.show()
df.describe().plot(kind = "Area",fontsize=15, figsize = (20,10), table = True,colormap="rainbow")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics")
plt.show()
sns.FacetGrid(df, hue="Generation", size=6).map(sns.kdeplot, "Population").add_legend()
plt.ioff() 
plt.show()
    
plt.figure(figsize=(10,7))
sns.stripplot(x="Year",y='Suicides100kPop',data=df)
plt.xticks(rotation=45)
plt.show()
col = ['Year', 'Population',
       'Suicides100kPop',
        'GdpPerCapitalMoney']
df1 = df[col]
print(df1.head())
y = df['SuicidesNo']
print(y.head())
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split

# devide the dataset into the 
train_x, test_x, train_y, test_y = train_test_split(df1,y, test_size = 0.2)
#print(train_x.shape, train_y.shape)
#print(test_y.shape, test_x.shape)
print(train_x.head())
model = DecisionTreeRegressor(random_state =0)
model.fit(train_x,train_y)
predict = model.predict(test_x)
#import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(np.arange(len(predict)),predict)
plt.title('predicted suicide rate')
plt.figure(2)
plt.plot(np.arange(len(predict)),test_y.values)
plt.title('actual suicide rate')
# accuracy score 
from sklearn.metrics import accuracy_score
b=accuracy_score(test_y, predict)
print("the accuracy score is"+str(b))
# mean absolute error 
def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

a=mape(test_y, predict)
print("the mean percentage absulate error is"+str(a))





