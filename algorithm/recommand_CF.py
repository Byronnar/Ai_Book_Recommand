# 为单个用户进行精准推荐 92行可以自定义用户ID

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 只显示error
tf.device('/gpu:0') # GPU训练

Book = pd.read_csv('data/BX-Books.csv', sep=None, error_bad_lines=False) # 读取文件
Book=Book[['ISBN', 'Book-Title']]
Book['index']=Book.index

Rating = pd.read_csv('data/BX-Book-Ratings.csv', sep=None, error_bad_lines=False)
Rating=Rating[:10000]

i=0
for x in set(Rating['User-ID']):
    Rating.loc[Rating['User-ID']==x,'userId']=i
    i+=1

Rating['userId']=Rating['userId'].astype(int)

Rating.columns=['User-ID', 'ISBN', 'rating','userId']   # 合并
ratings_df=pd.merge(Rating, Book, on = 'ISBN')

ratings_df = ratings_df[['userId','index','rating']]

userNo = ratings_df['userId'].max()+1
bookNo = ratings_df['index'].max()+1

rating = np.zeros((bookNo,userNo))
#标志位
flag = 0
#获取合并表中的列数
ratings_df_length = np.shape(ratings_df)[0]
#遍历矩阵，将书籍的评分填入表中
for index,row in ratings_df.iterrows():
    rating[int(row['index']), int(row['userId'])] = row['rating']
    flag += 1
    #print('processed %d, %d left' %(flag,ratings_df_length-flag))

record = rating > 0
record = np.array(record, dtype = int)

def normalizeRatings(rating, record):
    #获取书籍的数量m和用户的数量n
    m,n = rating.shape
    #rating_mean-书籍平均分   rating_norm-标准化后的书籍得分
    rating_mean = np.zeros((m,1))
    rating_norm = np.zeros((m,n))
    for i in range(m):
        idx = record[i,:]!=0
        rating_mean[i] = np.mean(rating[i,idx])
        rating_norm[i,idx] -= rating_mean[i]
    return rating_norm, rating_mean

rating_norm,rating_mean=normalizeRatings(rating,record)

rating_norm = np.nan_to_num(rating_norm)

rating_mean = np.nan_to_num(rating_mean)

num_features = 10

X_parameters = tf.Variable(tf.random_normal([bookNo, num_features],stddev = 0.35))

Theta_parameters = tf.Variable(tf.random_normal([userNo, num_features],stddev = 0.35))


#初始化模型
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

#加载模型
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './model/BookModel.ckpt')

Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])
# Current_X_parameters为用户内容矩阵，Current_Theta_parameters用户喜好矩阵
predicts = np.dot(Current_X_parameters,Current_Theta_parameters.T) + rating_mean


userId = 678 #此处用户可以自定义用户ID进行推荐


sortedResult = predicts[:, int(userId)].argsort()[::-1]
print ()

print("用户ID: ",int(userId))

idx = 0

print('为该用户推荐的评分最高的20部书籍是：'.center(80,'='))
for i in sortedResult:
    print('score: %.3f, book name: %s' % (predicts[i, int(userId)], Book.iloc[i]['Book-Title']))
    idx += 1
    if idx == 20: break



