import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


def load_and_preprocessing():
    df_train=pd.read_csv('train.csv')
    df_test=pd.read_csv('test.csv')
    df_train.shape,df_test.shape
    y_train=df_train.pop('SalePrice')
    # 便于处理，暂时concat训练集和测试集
    df=pd.concat((df_train,df_test),axis=0)
    
    # 对数据进行清洗
    total=df.isnull().sum().sort_values(ascending=False)
    percent=(df.isnull().sum()/len(df)).sort_values(ascending=False)
    miss_data=pd.concat([total,percent],axis=1,keys=['total','percent'])
    
    # 除去缺失率达40%以上的列
    df=df.drop(miss_data[miss_data['percent']>0.4].index,axis=1)
    
    # 无车库相关的属性则使用missing填充，车库建造时间的缺失使用1900填充
    garage_obj=['GarageType','GarageFinish','GarageQual','GarageCond']
    for garage in garage_obj:
       df[garage].fillna('missing',inplace=True)
    df['GarageYrBlt'].fillna(1900.,inplace=True)
    
    # 用missing标签表示没装修过，0表示没装修过的装修面积
    df['MasVnrType'].fillna('missing',inplace=True)
    df['MasVnrArea'].fillna(0,inplace=True)
    
    # 均值补齐LotFrontage列
    df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)
    
    # 还有部分少量的缺失值，用one-hotd转变离散值，然后均值补齐
    dummies_df=pd.get_dummies(df)
    mean_col=dummies_df.mean()
    dummies_df.fillna(mean_col,inplace=True)
    
    # 对数值进行标准化
    dummies_df['Id']=dummies_df['Id'].astype(str)
    a=dummies_df.columns[dummies_df.dtypes=='int64']
    b=dummies_df.columns[dummies_df.dtypes=='float64']
    a_mean=dummies_df.loc[:,a].mean()
    a_std=dummies_df.loc[:,a].std()
    dummies_df.loc[:,a]=(dummies_df.loc[:,a]-a_mean)/a_std
    b_mean=dummies_df.loc[:,b].mean()
    b_std=dummies_df.loc[:,b].std()
    dummies_df.loc[:,b]=(dummies_df.loc[:,b]-b_mean)/b_std
    df_train=dummies_df.iloc[:1460,:]
    
    # 分割训练集
    df_train_train, df_train_test, df_train_train_y, df_train_test_y=train_test_split(df_train,y_train,test_size=0.2)
    
    return df_train_train,df_train_test,df_train_train_y,df_train_test_y


if __name__ == '__main__':
    # 准备数据集
    df_train_train,df_train_test,df_train_train_y,df_train_test_y=load_and_preprocessing()
    
    
    from sklearn.model_selection import cross_val_score

    #对岭回归的正则化度进行调参
    alphas=np.logspace(-2,1.3,30)
    test_scores1=[]
    for alpha in alphas:
        clf=Ridge(alpha)
        scores1=np.sqrt(cross_val_score(clf,df_train_train,df_train_train_y,cv=5))
        test_scores1.append(1-np.mean(scores1))
    
    #从图中找出当正则化参数alpha为多少时，误差最小
    %matplotlib inline
    plt.plot(alphas,test_scores1,color='red')
    
    # 训练模型
    ridge=Ridge(alpha=10)
    ridge.fit(df_train_train,df_train_train_y)
    
    # 用均方误差判断模型好坏，越小越好
    pred=ridge.predict(df_train_test)
    print((((df_train_test_y-pred)**2).sum())/len(df_train_test_y))
    
    # 为预测结果标记Id并保存为csv
    pred_df=pd.DataFrame(pred,columns=['Price'])
    index=pd.DataFrame(df_train_test['Id']).reset_index(drop=True)
    pred_df=pd.concat([index,pred_df],axis=1)
    pred_df.to_csv('predict.csv', index=False)