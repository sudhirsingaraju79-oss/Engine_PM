# for data manipulation
import pandas as pd
import sklearn
## EDA
import matplotlib.pyplot as plt
import seaborn as sns
import math
from xgboost import XGBClassifier
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download
# format for EDA visualisation
sns.set(style="whitegrid", font_scale=1.1)
# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
# read data for Huggingface dataset space
DATASET_PATH = "hf://datasets/sudhirpgcmma02/Engine_PM/data/engine_data.csv"
df = pd.read_csv(DATASET_PATH)

## EDA univariate / bivariate / multivarite analysis
EDA_df(df)

################################# EDA ###########################################

def EDA_df(df):
  # ===============================================
  #   EDA FOR  FEATURES
  #
  # ===============================================
  features=[
      "Engine rpm",
      "Lub oil pressure",
      "Fuel pressure",
      "Coolant pressure",
      "lub oil temp",
      "Coolant temp"
  ]

  # -----------------------------
  # 1️ LOAD & BASIC INFORMATION
  # -----------------------------

  print("Shape:", df.shape)
  display(df.head(3))
  display(df.info())
  display(df.describe().T
          .style
          .format("{:.2f}")
          .background_gradient(cmap='Blues'))
  ## normatlise
  print(df['Engine Condition'].value_counts(normalize=True))

  # Hanlding missing
  print("missing values \n" ,df.isna().sum())

  summary=pd.DataFrame(
    {"Type":df.dtypes.values,
      "Mean":df.mean(numeric_only=True).round(2),
      "Max":df.max(numeric_only=True).round(2),
      "Min":df.min(numeric_only=True).round(2),
      "Missin (%)":df.isna().sum(),
      "count":df.count()}
  )
  print("########### Summary : Table #1 ###############\n",summary)


  # -----------------------------
  # 2️ MISSING VALUES
  # -----------------------------
  missing = df.isnull().sum().sort_values(ascending=False)
  if missing.any():
      mv = pd.DataFrame({
          "Missing Count": missing[missing > 0],
          "Missing %": (missing[missing > 0]/len(df)*100).round(2)
      })
      display(mv)
      plt.figure(figsize=(12,5))
      ax=sns.barplot(x=mv.index[:20], y="Missing Count", data=mv, color='steelblue')
      for container in ax.containers:
          ax.bar_label(container,label_type='center')
      ax.set_xticklabels(['Normal','Preventive Maintenance required'])
      plt.xticks(rotation=90)
      plt.title("Features Missing Values")
      plt.show()
  else:
    print(" No missing values in the dataset")
  # -----------------------------
  # 3️ SPLIT FEATURE TYPES
  # -----------------------------
  num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

  print(f" #####################   Numeric Features: {len(num_cols)} ####################")

  # -----------------------------
  # 4️ Column char (Numeric)
  # -----------------------------
  print("\n📦 Bar Charts for Top Categorical Features")
  i=0
  for col in num_cols[:5]:
      plt.figure(figsize=(8,4))
      ax=sns.barplot(x='Engine Condition',y=col, data=df, estimator='mean', palette='viridis')
      for container in ax.containers:
          ax.bar_label(container,label_type='center',fmt='%.2f')
      plt.title(f"Frequency Distribution: {col} | chart # {i+1}")
      plt.legend(
      title='Engine condition',
      labels=['Normal (0)','Preventive Maintenance required (1)']
      )
      plt.tight_layout()
      plt.show()
      i+=1

  # -----------------------------
  # 5️ COLUMN (BAR) CHARTS (Categorical)
  # -----------------------------


  print("\n###################   Histograms for Numeric Features ##################################")
  i+=1
  df_chart=df.melt(
    id_vars="Engine Condition",
    value_vars=features,
    var_name="Sensor",
    value_name="value"
  )

  plt.figure(figsize=(18,5))
  ax=sns.barplot(x="Sensor",y="value",hue="Engine Condition",estimator="mean",errorbar=None,data=df_chart)
  for container in ax.containers:
          ax.bar_label(container,label_type='center',fmt='%.2f')
  #ax.set_xticklabels(['Normal','Breakdown'])
  ax.set_ylabel("Value (Actual)")
  plt.title(f"Sensor vs Engine Condition | Chart {i}")

  plt.legend(
      title='Engine condition',
      labels=['Normal (0)','Preventive Maintenance required (1)']
      )
  plt.tight_layout()
  #plt.show()

  df_stk=df.copy()
  df_stk[features]=StandardScaler().fit_transform(df_stk[features])

  df_long=df_stk.melt(
    id_vars="Engine Condition",
    value_vars=features,
    var_name="Sensor",
    value_name="value"
  )
  plt.figure(figsize=(18,5))
  i+=1
  ax=sns.barplot(x="Sensor",y="value",hue="Engine Condition",estimator="mean",ci=None,data=df_long)

  for container in ax.containers:
          ax.bar_label(container,label_type='center',fmt='%.2f')
  handles,_=ax.get_legend_handles_labels()
  ax.set_ylabel("Value (Normalised 0-1)")
  plt.title(f"Sensor vs Engine Condition  | Chart {i}")
  plt.xticks(rotation=90)
  plt.legend(
      title='Engine condition',
      labels=['Normal (0)','Preventive Maintenance required (1)']
      )
  plt.tight_layout()
  plt.show()



  # -----------------------------
  # 6️ LINE CHART (Trend View)
  # -----------------------------
  print("\n📈 Line Chart for Numeric Feature Trends")
  i+=1
  plt.figure(figsize=(12,6))
  df1=df.reset_index()
  df1['step']=range(len(df))
  ax=sns.lineplot(
      data=df1,
      x='Engine rpm',
      y='Engine Condition',
      color="steelblue",
      label="Engine Condition"
  )

  sns.scatterplot(
      data=df1[df1['Engine Condition']==1],
      x='Engine rpm',
      y='Engine Condition',
      color='red',
      marker="X",
      s=80,
      label="Preventive Maintenance "
  )
  plt.xlabel("Breakdonw obsrvation")
  plt.ylabel("Engine condition")
  plt.title(f"Engine Condition Trend | chart {i}")
  plt.legend(
      title='Engine condition',
      labels=['Normal (0)','Preventive Maintenance required (1)']
      )
  plt.tight_layout()
  plt.show()
  # -----------------------------
  # 7️ BOX PLOTS (Outlier View)
  # -----------------------------
  print("\n📦 Boxplots for Numeric Features")
  i+=1
  plt.figure(figsize=(16,8))
  ax=sns.boxplot(data=df[num_cols[:10]], orient='h', palette='coolwarm')
  plt.title(f"Boxplot Numeric Features | Chart {i}")
  plt.show()

  # -----------------------------
  # 8️ STACKED COLUMN CHART
  # -----------------------------
  print("\n🧱 Stacked Bar Chart (Numeric grouped by Categorical Feature)")
  trg="Engine Condition"
  i+=1
  #if len(num_cols) > 0:
     # cat = cat_cols[0]
  grouped = df.groupby(trg)[num_cols].mean().head(10)
  ax=grouped.T.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Spectral')
  for container in ax.containers:
      ax.bar_label(container,label_type='center',fmt='%.2f')
  plt.title(f"Stacked Mean of {num_cols} | chart {i}")
  plt.ylabel("Mean Value")
  plt.legend(
      title='Engine condition',
      labels=['Normal (0)','Preventive Maintenance required (1)']
      )
  plt.show()

  # -----------------------------
  # 9️ PIE CHARTS (Numeical Composition)
  # -----------------------------
  print("\n🥧 Pie Charts for Features")
  num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
  i+=1
  for col in num_cols:
    uniq=df[col].nunique()
    plt.figure(figsize=(5,5))
    if uniq <= 10:
      cnt=df[col].value_counts()
      label=cnt.index
      wedg, txt, autotxt = plt.pie(cnt, labels=cnt.index, autopct='%1.1f%%', startangle=90)
      plt.legend(wedg,
                 ['Normal (0)','Preventive Maintenance required (1)'],
                 title='Engine Condition',
                 loc='center left',
                 bbox_to_anchor=(1,0.5)
      )
      plt.title(f"Pie Chart of {col} | chart {i}")
      plt.axis('equal')
      plt.show()
      i+=1

  # -----------------------------
  # 10 CORRELATION MATRIX + TABLE
  # -----------------------------
  print("\n🧩 Correlation Analysis")
  corr = df[num_cols].corr()
  i+=1
  plt.figure(figsize=(12,10))
  sns.heatmap(corr, cmap='coolwarm', center=0,annot=True,fmt =".2f" )
  plt.title(f"Correlation Heatmap | Chart {i}")
  plt.show()

  # Top correlated pairs
  corr_pairs = corr.unstack().sort_values(ascending=False)
  corr_pairs = corr_pairs[corr_pairs < 1]  # remove self correlation
  top_corr = corr_pairs.head(20).to_frame("Correlation")
  display(top_corr.style.background_gradient(cmap='RdYlGn'))

  #############################################################
  # 11 Histogram
  #
  ##############################################################
 # target distribution
  num_fea = df.select_dtypes(include=["int64","float64"]).columns
  nf=len(num_fea)
  col=4
  i+=1
  rows=math.ceil(nf/col)

  plt.figure(figsize=(20,rows*4))

  fig, axes = plt.subplots (
        rows , col,
        figsize=(22,rows *5),
        constrained_layout=True
    )
  axes =  axes.flatten()

  for i,col in enumerate(num_fea):
    #plt.subplot(len(num_fea)//3+1,3,i)
    #plt.subplot(rows,col,i)
    ax=axes[i]
    sns.histplot(df[col],kde=True,bins=30,
                  ax= ax
                  )
    ax.set_title(col, fontsize=12)
    ax.tick_params(axis='both',labelsize=9)

  for j in range( i+1 , len(axes)):
    fig.delaxes(axes[j])

  plt.title(f"Histogram for distribution of features | chart {i}")
  plt.tight_layout()
  plt.show()



  ###############################################################################
  #  12 PAIR plot                                                                #
  ###############################################################################
  i+=1
  g=sns.pairplot(df[features+ ["Engine Condition"]],hue="Engine Condition",diag_kind="kde",corner=True)
  g.fig.suptitle(f"Feature interaction char {i}")
  n_lbl=['Normal (0)','Preventive Maintenance required (1)']
  for t,l in zip(g._legend.texts,n_lbl):
    t.set_text(l),
  plt.show()

  ###############################################################################
  ## 13 Priciple componenet analysis                                            #
  ###############################################################################

  x=df[features]
  i+=1
  y=df["Engine Condition"]

  scaler=StandardScaler()
  x_scaled=scaler.fit_transform(x)
  pca=PCA(n_components=2)
  x_pca=pca.fit_transform(x_scaled)
  plt.figure(figsize=(5,5))
  sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=y,alpha=0.6)

  plt.title(f"PCA of Features for Engine Condition | chart {i}")
  plt.legend(
      title='Engine condition',
      labels=['Normal (0)','Preventive Maintenance required (1)']
  )
  plt.show()

#Features naming standardisation for easy handling
df.columns = (df.columns
                   .str.strip()
                   .str.replace(" ","_")
                   .str.replace(r"[^\w]","_",regex=True)
  )



# Targe varaible intialisation
target_col = 'Engine_Condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sudhirpgcmma02/Engine_PM",
        repo_type="dataset",
    )
print("Dataset after split  loaded successfully to Huggingface.....")
