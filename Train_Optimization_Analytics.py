import folium
from geopy.geocoders import Nominatim,Photon
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statistics
import os
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train_data.csv')


# 1 Data Cleaning
# 1.1 Missing Values
print("1.1 The sum of missing values")
print((data.isna() | data==0).sum())
print("*" * 50)

# 1.2 Detect Outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=data.iloc[:,:-1].select_dtypes(include=['float64', 'int64']))
plt.title('Boxplots of Numerical Variables to Detect Outliers')
plt.xticks(rotation=45)
plt.show()

# 1.3 Duplicated Values

duplicate_rows = data[data.duplicated()]
print("1.3 The situation of duplicated values")
if len(duplicate_rows) == 0:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found:")
    print(duplicate_rows)
print("*" * 50)

# Data cleaning steps
# Drop columns with all missing values and unnecessary columns
cleaned_data = data.drop(columns=['Itinerary Code', 'Unnamed: 0'])

cleaned_data.dropna(inplace=True)


# Convert date fields to datetime format
cleaned_data['Travel Start Date'] = pd.to_datetime(cleaned_data['Travel Start Date'])
cleaned_data['Travel End Date'] = pd.to_datetime(cleaned_data['Travel End Date'])

cleaned_data['Mileage Per ticket'] = cleaned_data['Mileage'] / cleaned_data['Segment Count']
cleaned_data['Ticket Expense Per ticket'] = cleaned_data['Ticket Expense'] / cleaned_data['Segment Count']
cleaned_data['Carbon Emission Per ticket'] = cleaned_data['Carbon Emission'] / cleaned_data['Segment Count']

# Extract day of the week and hour from the Travel Start Date
cleaned_data['Day of Week'] = cleaned_data['Travel Start Date'].dt.day_name()


train_data = cleaned_data
cleaned_data.to_csv('cleaned_traindata.csv')
# Display the cleaned data summary and check types
cleaned_data.info()

sta2city_raw = pd.read_csv('station.csv',encoding='GBK')
sta2city = {}
for i in sta2city_raw['station']:
    sta2city[i.split('→')[0].strip()] = i.split('→')[-1].strip()
sta2city['Paris Gare De L&#39;Est'] = 'Paris'

def func(a):
    return sta2city[a.strip()]
train_data['origin_city'] = train_data.Origin.apply(lambda x: sta2city[x.strip()])
train_data['destination_city'] = train_data.Destination.apply(lambda x: sta2city[x.strip()])
train_data['route_city'] = train_data['origin_city'] + "->" + train_data['destination_city']
train_data.to_csv('final_traindata.csv')
ori2des = set(zip(list(train_data.Origin.apply(lambda x: sta2city[x.strip()])),list(train_data.Destination.apply(lambda x: sta2city[x.strip()]))))



# 2.EDA
# 2.1 Statistical Summary
print("2.1 Statistical Summary")
train_data.describe()


# 2.2 KDE
def Init_plot(x, plot_data, path='Figures/init_plot.png'):
    # Draw histograms and KDEs using commands suited to
    kdeAxes = sns.kdeplot(plot_data, color="black", label="Kernel Density")
    sns.histplot(plot_data, stat="density", color="lightsteelblue", label="Histogram")

    # Finally, add a rug plot
    sns.rugplot(plot_data, label="Rug Plot")

    # Add labels
    plt.xlabel(x)
    plt.ylabel('Estimated Density')

    # Add tick marks abd set the limits of the axes
    #     plt.xticks(range(3,10))
    #     plt.xlim([3, 9])
    plt.legend()
    plt.tight_layout()

    # Save a Pdata version, then display the result here too


#     plt.savefig(path)
#     plt.show()

# Summary statistics
def sum_sta(x, sum_data, long):
    #     plt.figure(figsize=(12,8))
    d_mean = np.mean(sum_data)
    d_var = np.var(sum_data)
    ss = np.sqrt(np.var(sum_data))

    # skewness
    skewness = sp.stats.moment(sum_data, 3) / (ss ** 3)

    # kurtosis
    kurtosis = (sp.stats.moment(sum_data, 4) / (ss ** 4)) - 3

    # Mode
    d_mode = statistics.mode(sum_data)

    # Median
    d_median = statistics.median(sum_data)

    Init_plot(x, sum_data)
    # Get ready to plot vertical lines
    xx = np.ones(2)
    yy = np.array([0, long])

    # Add variouosly dashed vertical lines for the three
    # measures fo central tendency
    plt.plot(d_mean * xx, yy, '--b', label=f'Mean={round(d_mean, 2)}')
    plt.plot(d_median * xx, yy, '-.r', label=f'Median={round(d_median, 2)}')
    plt.plot(d_mode * xx, yy, ':m', label=f'KDE Mode={round(d_mode, 2)}')
    plt.legend()


def eda_plot(col_name, x, y, n, long):
    plot_data1 = train_data[col_name]
    plt.subplot(x, y, n)
    plt.title(f'{col_name}')
    sum_sta(col_name, plot_data1, long)


plt.figure(figsize=(20, 7))
cnt = 1
longlist = [0.01, 0.018, 0.25]
for i in ['Mileage Per ticket', 'Ticket Expense Per ticket', 'Carbon Emission Per ticket']:
    eda_plot(i, 1, 3, cnt, longlist[cnt - 1])
    cnt += 1

plt.show()

#2.3 pair plot
sns.pairplot(train_data.drop(['Employee ID','Segment Count','Mileage','Carbon Emission','Ticket Expense'],axis=1),diag_kind='kde')
plt.show()



# 3. visualize map
route_cnt = train_data[train_data.origin_city != train_data.destination_city].groupby("route_city")['Segment Count'].sum().sort_values(ascending=False)
route_cnt[route_cnt > 10]

compare_route1 = train_data[train_data.route_city.isin(list(route_cnt[route_cnt >= 10].keys()))]
ori2des_cnt10 = set(zip(list(compare_route1.Origin.apply(lambda x: sta2city[x.strip()])),list(compare_route1.Destination.apply(lambda x: sta2city[x.strip()]))))
geolocator = Nominatim(user_agent="geoapiHAPPY")
# geolocator = Photon(user_agent="measurements")


def get_location(city):
    if city in ['Hayes','Stockton','Rose Hill','Newport','Chesterfield','Tamworth','Ebbsfleet','Lancaster','Newark','Livingston','Ware','Hertford']:
        city += ",UK"
    location = geolocator.geocode(city)
    return location.latitude, location.longitude


m = folium.Map(location=[54.7023545, -3.2765753], zoom_start=6)  # 英国的大致中心点


cnt = 0
for i in tqdm(ori2des_cnt10):

    cnt += 1
    if i[0] == 'Unknown' or i[1] == 'Unknown': continue
    try:
        time.sleep(1)
        origin = get_location(i[0])
        destination = get_location(i[1])

        folium.Marker(origin, popup=i[0]).add_to(m)
        folium.Marker(destination, popup=i[1]).add_to(m)

        folium.PolyLine(locations=[origin, destination], color='red').add_to(m)
    except Exception as e:
        print(e)

m.save('uk_train_routes_cnt10_new.html')



# 4. Traffic pattern analysis
# 计算每日的旅行次数
day_of_week_counts = train_data['Day of Week'].value_counts().sort_values(ascending=False)


plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='purple')
plt.title('Travel Frequency by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Travels')
plt.grid(True)
plt.show()

origin_counts =  train_data.groupby('Origin')['Segment Count'].sum().sort_values(ascending=False)[:10]
destination_counts = train_data.groupby('Destination')['Segment Count'].sum().sort_values(ascending=False)[:10]

route_counts = train_data.groupby('Route')['Segment Count'].sum().sort_values(ascending=False)[:10]

fig, axs = plt.subplots(3, 1, figsize=(14, 18))

axs[0].bar(origin_counts.index, origin_counts.values, color='skyblue')
axs[0].set_title('Travel Frequency by Origin')
axs[0].set_xlabel('Origin')
axs[0].set_ylabel('Number of Travels')
axs[0].tick_params(axis='x', rotation=90)

axs[1].bar(destination_counts.index, destination_counts.values, color='lightgreen')
axs[1].set_title('Travel Frequency by Destination')
axs[1].set_xlabel('Destination')
axs[1].set_ylabel('Number of Travels')
axs[1].tick_params(axis='x', rotation=90)

axs[2].bar(route_counts.index, route_counts.values, color='salmon')
axs[2].set_title('Most Busy Routes')
axs[2].set_xlabel('Route')
axs[2].set_ylabel('Number of Travels')
axs[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show(), origin_counts, destination_counts, route_counts

