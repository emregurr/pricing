import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.stats.api as sms
import itertools

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:.4f}'.format

pricing =pd.read_csv("Datasets/pricing.csv", sep=";")
df=pricing.copy()
df.head()

def information(dataframe):
    print("shape" , dataframe.shape)
    print("Index", dataframe.index)
    print("Columns", dataframe.columns)
    print("Null", dataframe.isnull().values.any())
    print("Count of Null" , df.isnull().sum())

information(df)

# shape (3448, 1)
# Index RangeIndex(start=0, stop=3448, step=1)
# Columns Index(['category_id;price'], dtype='object')
# Null False
# Count of Null category_id;price    0
# dtype: int64

df.category_id.value_counts()

# 489756    1705
# 874521     750
# 361254     620
# 326584     145
# 675201     131
# 201436      97
# Name: category_id, dtype: int64

df.category_id.unique()
# array([489756, 361254, 874521, 326584, 675201, 201436], dtype=int64)


def analysis(dataframe, category, target, alpha):
    AB = pd.DataFrame()
    combin = list(itertools.combinations(df.category_id.unique(), 2))
    print("-" * 20, "Grup Karşılaştırmaları | alpha güven katsayısı :", alpha, "-" * 20, )
    for i in range(0, len(combin)):
        grA = dataframe[dataframe[category] == combin[i][0]][target]
        grB = dataframe[dataframe[category] == combin[i][1]][target]

        # TESTLER
        # NORMALLİK VARSAYIMI
        normA = shapiro(grA)[1] < alpha
        normB = shapiro(grB)[1] < alpha

        # Ho: Seri normal dağılıyor. Ho > 0.05
        # H1: Seri normal dağılmıyor. Ho < 0.05

        if (normA == False) & (normB == False):
            # İki dağılım da normal. Levene Testine Geçilebilir.
            # Levene Testi. Varyanslar homojen mi?

            levene = stats.levene(grA, grB)[1] < alpha
            # Ho: Varyanslar homojen. Ho > 0.05
            # H1: Varyanslar homojen değil. Ho < 0.05

            if levene == False:
                # Varyanslar homojen

                ttest = stats.ttest_ind(grA, grB, equal_var=True)[1]
                # Ho: M1=M2 Aralarında fark yok. Ho > 0.05
                # H1: M1!=M2 Aralarında fark var. Ho < 0.05
            else:
                # Varyanslar homojen değil, welch testi

                ttest = stats.ttest_ind(grA, grB, equal_var=False)[1]
                # Ho: M1=M2 Aralarında fark yok. Ho > 0.05
                # H1: M1!=M2 Aralarında fark var. Ho < 0.05

        else:  # Dağılımlardan en az birisi normal değil. non - parametric test

            ttest = stats.mannwhitneyu(grA, grB)[1]
            # Ho: M1=M2 Aralarında fark yok. Ho > 0.05
            # H1: M1!=M2 Aralarında fark var. Ho < 0.05

        # Sonuç
        temp = pd.DataFrame({"Grup Karşılaştırması": [ttest < alpha],
                             "p-value": ttest,
                             "GroupA Mean": [grA.mean()], "GroupB Mean": [grB.mean()],
                             "GroupA Median": [grA.median()], "GroupB Median": [grB.median()],
                             "GroupA Count": [grA.count()], "GroupB Count": [grB.count()]}, index=[combin[i]])

        temp["Grup Karşılaştırması"] = np.where(temp["Grup Karşılaştırması"] == True, "Fark Var", "Fark Yok")
        temp["Test Tipi"] = np.where((normA == False) & (normB == False), "Parametrik", "Non-Parametrik")

        AB = pd.concat([AB, temp[
            ["Test Tipi", "Grup Karşılaştırması", "p-value", "GroupA Mean", "GroupB Mean", "GroupA Median",
             "GroupB Median", "GroupA Count", "GroupB Count"]]])
    return AB


# İtemin Fiyatı Kategorilere Göre Farklılık Gösteriyor Mu?

AB = analysis(df,"category_id","price",0.05)
AB


# Aralarında Fark Olan Gruplar


ABFARKVAR=AB[AB["Grup Karşılaştırması"]=="Fark Var"]


# Aralarında Fark Olmayan Gruplar

ABFARKYOK=AB[AB["Grup Karşılaştırması"]=="Fark Yok"]



# Gruplar normallik varsayımını sağlamadığından dolayı outlier'lara bakabiliriz


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print("Total Data Size:",dataframe.shape[0])
            print(col, ":", number_of_outliers, "outlier value")
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    #return variable_names


has_outliers(df, ["price"])

# Total Data Size: 3448
# price : 77 outlier value

def remove_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return df_without_outliers


df2 = remove_outliers(df,"price")

information(df2)

# shape (3371, 2)
# # Index Int64Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
# #             ...
# #             3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447],
# #            dtype='int64', length=3371)
# # Columns Index(['category_id', 'price'], dtype='object')
# # Null False
# # Count of Null category_id    0
# # price          0
# # dtype: int64

# Outlier Sildikten Sonraki Rapor Sonuçları

AB2 = analysis(df2,"category_id","price",0.05)
AB2


# Kategoriler hala normal dağılmıyor fakat,
# outlier gözlemleri kaldırdıktan sonra kategoriler arasındaki ortalamalar birbirine yakın değerlerden oluştu.

# İstatistiksel olarak aralarında anlamlı bir farklılık olan gruplar Ho < 0.05

AB2FARKVAR=AB2[AB2["Grup Karşılaştırması"] == "Fark Var"]


# İstatistiksel olarak aralarında anlamlı bir farklılık olmayan gruplar. Ho > 0.05

AB2FARKYOK=AB2[AB2["Grup Karşılaştırması"] == "Fark Yok"]


Fark_var = AB[AB["Grup Karşılaştırması"] == "Fark Var"].index
Fark_var = pd.DataFrame({"Difference": Fark_var})
Fark_var


Fark_yok = AB[AB["Grup Karşılaştırması"] == "Fark Yok"].index
Fark_yok = pd.DataFrame({"No Difference": Fark_yok})
Fark_yok



# Kendi aralarında benzer olan kategoriler;
#
# 361254
# 874521
# 675201
# 201436
# Diğerlerinden farklı olan kategoriler;
#
# 489756
# 326584
# Görüldüğü gibi itemin fiyatı kategorilere göre farklılık göstermektedir



# İtem Fiyatı Ne Olmalı?

df2.groupby(["category_id"]).agg({"price":["mean","median","min","max","count"]})

#                           price
#                mean  median     min      max count
# category_id
# 201436      36.1755 33.5347 30.0000  74.4529    97
# 326584      35.6932 31.7060 30.0000 103.3825   144
# 361254      35.4773 34.4565 30.0000 111.5184   615
# 489756      43.6040 35.3991 10.0000 186.7400  1658
# 675201      37.4436 33.7259 30.0000  92.8926   129
# 874521      39.2732 34.2036 10.0000 187.4451   728

# Tüm Kategorilere Aynı Fiyat İçin Fiyat Katalogu

# Bu kısımda kategoriler arasındaki farka bakmaksızın bütün kategoriler için;
#
# Güven aralığının en düşük limit değeri:
# Güven aralığının orta değeri:
# Güven aralığının en yüksek limit değeri:
# Kategorilerin mean değerlerinin ortalamaları
# Kategorilerin median değerlerinin ortalamaları
# Fiyat kataloguna yarımcı fonksiyonlar ile eklenecek ve sonucunda bu fiyatlara göre olan satışlar
# simule edilecektir

# bu fonksiyon kategoriler arasındaki ortalamaların veya medianların ortalamasını alıyor
def fiyat_belirleme(dataframe, category="category_id", target="price", method="median"):
    toplam = 0
    if method == "mean":
        for i in dataframe[category].unique():
            toplam = toplam + dataframe[dataframe[category] == i][target].mean()
    if method == "median":
        for i in dataframe[category].unique():
            toplam = toplam + dataframe[dataframe[category] == i][target].median()
    return (toplam / dataframe[category].nunique())


# Fiyat: Tüm Kategorilerin Fiyat Listeleri

# Bu fonksiyon ilgili dataframein güven aralıklarını, mean ve median değerlerini bir listeye ekliyor. bu değerler frekans sayıları ile çarpılıp gelir tahmini yapılacak.
def price_list(dataframe,target,liste):
    lower, upper = sms.DescrStatsW(dataframe[target]).tconfint_mean()
    mid = (lower + upper) / 2
    mean = fiyat_belirleme(dataframe, method="mean")
    median = fiyat_belirleme(dataframe, method="median")
    liste.append(lower)
    liste.append(mid)
    liste.append(upper)
    liste.append(mean)
    liste.append(median)
    return liste


same_category = []
price_list(df2, "price",same_category)

# [39.78385949600828, = En düşük limit
#  40.39865162533902, = Orta limit
#  41.01344375466976, = üst limit
#  37.94444677595247, = Mean degeri
#  # 33.837631079516676] = Median degeri


# Bu fonksiyon belirtilen eşik fiyatlarına göre fiyat simülasyonunu çıkarıyor
def gelir_upd(dataframe,target,th):
    frekans = len(dataframe[dataframe[target]>=th]) #güven aralığının eşik noktasından yüksek ve eşit olan satın almaların sayısı
    gelir = frekans * th #gelir hesabı
    return gelir

# TÜM KATEGORİLERE AYNI FİYAT SİMÜLASYONU


def satis_simulasyon(fiyat):
    count=1
    for th in fiyat:
        print(count,"SENARYO | %.4f" %th ,"satış fiyatı için gelir tahmini")
        x = gelir_upd(df2,"price",th)
        print(x)
        print("\n")
        count += 1


satis_simulasyon(same_category)

# 1 SENARYO | 39.7839 satış fiyatı için gelir tahmini
# 34532.39004253519
# 2 SENARYO | 40.3987 satış fiyatı için gelir tahmini
# 34177.25927503681
# 3 SENARYO | 41.0134 satış fiyatı için gelir tahmini
# 33138.862553773164
# 4 SENARYO | 37.9444 satış fiyatı için gelir tahmini
# 35781.61330972318
# 5 SENARYO | 33.8376 satış fiyatı için gelir tahmini
# 68013.63846982853


# Sonuç olarak en iyi gelir:
# kategorilerin medianlarının ortalaması (33.8376) alınarak yapılan satış oldu. 5. SENARYO. GELİR : 68013.64

# Güven aralığına göre en iyi sonucu veren satış ise;
# güven aralığının en alt limiti olan (39.7839) satış fiyatıyla yapılan satış oldu.1. SENARYO GELİR: 34532.39
