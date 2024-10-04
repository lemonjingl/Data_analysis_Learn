import matplotlib.pyplot as plt
import numpy as np
def plot(kmeans_model=None,columns=None):
    '''
    kmeans_model:表示的是模型的聚类个数
    columns:表示的是各属性的名称
    None:作为两者的初始化赋值
    '''
    plt.figure(figsize=(11,11))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.style.use('ggplot')
    N=len(columns)
    angles=np.linspace(0,2*np.pi,N,endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))

    Linecolor=['r-','o-','g--','b-.','p:']
    # 绘图
    feature=list(columns)+['L']

    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(1,1,1,polar=True)

    ax.set_thetagrids(angles * 180 /np.pi,feature)# 设置极坐标角度网格线
    ax.set_ylim(kmeans_model.cluster_centers_.min(),kmeans_model.cluster_centers_.max())
    plt.title('聚类属性分布')
    ax.grid(True)
    lab=[]
    for i in range(len(kmeans_model.cluster_centers_)):
        values=kmeans_model.cluster_centers_[i]
        values=np.concatenate((values,[values[0]]))
        ax.plot(angles,values,Linecolor[i],linewidth=2)
        ax.fill(angles,values,alpha=0.25)
        lab.append('类别'+str(i))
    plt.legend(lab)
    plt.show()
