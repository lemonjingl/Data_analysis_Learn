import pandas as pd
import numpy as np 

# 编写UCF推荐算法
def UCF_rec(train_df,test_df,s_data,N):
    rec_pre=pd.DataFrame()
    for u in test_df.index:# 获取每一个测试数据用户
        # 获取推荐电影
        try:
            s_users=s_data.loc[u].dropna().sort_values().index[int('-'+str(N)):]# 推荐相似度前N个用户
            s_items=[]
            for v in s_users:
                s_items+=list(trian_df.loc[v,train_df.loc[v,:]!=0].index)# 获取当前推荐用户的所有偏好动漫
            u_items=test_df.loc[u,test_df.loc[u]!=0].index# 获取当前用户的偏好动漫
            rec_items=list(set([i for i in s_items if i not in u_items]))

            # 获取预测评分
            s_uv=s_data.loc[u,list(s_users)].values # 获取与用户的相似度sim
            train_df_tmp=train_df.loc[s_users,rec_items].values # 获取相似用户对动漫的评分
            r_m=np.nanmean(train_df.loc[s_users],axis=1)
            U_array=(train_df_tmp-(r_m.reshape(-1,1)*np.ones((N,train_df_tmp.shape[1]))))*s_uv.reshape(-1,1)
            U=np.nansum(U_array,axis=0)
            s_uv_tmp=s_uv.reshape(-1,1)*np.ones((len(s_users),train_df_tmp.shape[1]))*(train_df_tmp/train_df_tmp)
            D=np.nansum(s_uv_tmp*r_m.reshape(-1,1),axis=0)
            D=np.nansum(s_uv_tmp,axis=0)
            p=np.mean(r_m)+U/D

            rec_pre_temp=pd.DataFrame(columns=rec_items)
            rec_pre_temp.loc[u,:]=p# 行为预测数据用户，列为推荐的动漫，值为预测评分
            rec_pre=pd.concat([rec_pre,rec_pre_temp])
        except:
            pass
    return rec_pre