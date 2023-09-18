import eda as hlp
import numpy as np
import pandas as pd
import random
import warnings
import time
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder



from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=FutureWarning)

## Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

df_1 = pd.read_csv("pythonProject/Scoutium-220805-075951/scoutium_attributes.csv", sep=";")
df_2 = pd.read_csv("pythonProject/Scoutium-220805-075951/scoutium_potential_labels.csv", sep=";")

df_1.head()
df_2.head()

df_1.shape

## Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)


df = pd.merge(df_1, df_2, how="inner", on=["task_response_id", "match_id", "evaluator_id", "player_id"])

df.head()

df.shape


## Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df[~(df["position_id"] == 1)]
df.head()


## Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)


df = df[~(df["potential_label"] == "below_average")]

df.shape


## Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

### Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve
# değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.


df = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns=["attribute_id"], values="attribute_value")
df


### Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

df = df.reset_index()
df.head()

df.columns.name = None
df

df.columns = df.columns.astype(str)
df.head()


## Adım6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df, "potential_label")

df

## Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız

def grab_col_names(dataframe,cat_th=10,car_th=20):
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=="object"]
    num_but_cat=[col for col in dataframe.columns if dataframe[col].nunique()<cat_th and dataframe[col].dtypes!="object"]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes=="object"]
    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes!="object"]
    num_cols=[col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print(f"cat_but_car:{len(cat_but_car)}")
    # cat_cols + num_cols+ cat_but_car =değişken sayısı
    # num_but_cat cat_cols un içinde zaten, sadece raporlama için verilmiş
    return cat_cols,cat_but_car,num_cols,num_but_cat

cat_cols,cat_but_car,num_cols,num_but_cat = grab_col_names(df)

num_cols = num_cols + ['4324', '4328', '4352', '4357', '4423']
num_cols
df.info()
## alternatif : num_cols = [col for col in player_df.columns if col not in ["player_id", "position_id", "potential_label"]]

## Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

## Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.
# (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)


y = df["potential_label"]
X = df.drop(["player_id", "potential_label"], axis=1)


print(f"The Shape of X is --> {X.shape}")
print(f"The Shape of y is --> {y.shape}")

models = [
    ("LR", LogisticRegression()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("RF", RandomForestClassifier()),
    ("GBM", GradientBoostingClassifier()),
    ("XGB", XGBClassifier()),
    ("LGBM", LGBMClassifier()),
    ("CAT", CatBoostClassifier(verbose=False)),
]


for name, model in models:
    print(name)
    start = time.time()
    for score in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_validate(model, X, y, cv=10, scoring=score)
        print(score, ":", scores["test_score"].mean())
    end = time.time()
    print("runtime: %.2f s"%(end - start))
    print("-" * 20)



## Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)




