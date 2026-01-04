import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import uvicorn
import hashlib
from schema import PostGet, Response

app = FastAPI()

logger.add("experiment.log", rotation="10 MB", level="INFO")

engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

user_data = pd.read_sql('SELECT * FROM public.user_data', con = engine)

preprocess_posts = pd.read_sql('SELECT * FROM public.preprocess_posts', con = engine)
preprocess_posts_new = pd.read_sql('SELECT * FROM public.kazakov_mta3669_features_dl', con = engine)
posts = pd.read_sql('SELECT * FROM public.post_text_df', con = engine)
posts.set_index('post_id', inplace=True) 
# Обеспечивает 100 % пользователей, присутствующих в user_data 
def load_features() -> pd.DataFrame:
    feed_data = batch_load_sql("""
        SELECT *
        FROM public.feed_data
        WHERE action='like'
        limit 9000000""")
    return feed_data
features_data = load_features() 

def get_exp_group(user_id: int) -> str:
    hash_val = int(int(hashlib.sha256((str(user_id) + 'my_salt').encode()).hexdigest(), 16) % 100)
    temp_exp_group = hash_val % 100
    return 'control' if temp_exp_group <= 50 else 'test'

def get_model_path(exp_group: str) -> str:
    """Полный путь к модели с учетом среды и группы"""
    #if os.environ.get("IS_LMS") == "1": только для чекера
    if exp_group == 'test':
        MODEL_PATH = 'Models/model_test.cbm'
        return MODEL_PATH
    elif exp_group == 'control':
        MODEL_PATH = 'Models/model_control.cbm'
        return MODEL_PATH
    else:
        raise ValueError(f"Unknown experiment group: {exp_group}")
    
def load_models(exp_group: str):
    model_path = get_model_path(exp_group)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

model_control = load_models("control")
model_test = load_models("test")

#Функция классического ML
def recommendations_ML(id: int, time: datetime, limit: int):
    user_view_posts = features_data[features_data['user_id'] == id]['post_id'].unique() 
    select_posts = preprocess_posts[~preprocess_posts['post_id'].isin(user_view_posts)]
    user_init = user_data[user_data['user_id'] == id]
    select_posts_use = select_posts.copy() 
    for col in user_init.columns:
        select_posts_use = select_posts_use.assign(**{col: user_init[col].iloc[0]})
    select_posts_use = select_posts_use.set_index(['user_id', 'post_id'])
    viewing_posts = features_data[features_data['user_id'] == id]['target'].count()
    like_posts = features_data[(features_data['user_id'] == id) & (features_data['target'] == 1)]['target'].count()
    like_frequency = like_posts / viewing_posts if viewing_posts > 0 else 0
    user_dates = features_data[features_data['user_id'] == id]['timestamp']
    min_date = user_dates.min()
    select_posts_use['target'] = like_frequency
    select_posts_use['min_date'] = min_date
    categorical_features = ['country', 'city', 'os', 'source']
    C = 0.006
    for col in categorical_features:
        if col in ['os', 'source']:
            if select_posts_use[col].nunique() == 2: 
                dummies = pd.get_dummies(select_posts_use[col], prefix=col, dtype=int, drop_first=True)
                select_posts_use = pd.concat([select_posts_use, dummies], axis=1)
                select_posts_use = select_posts_use.drop(col, axis=1)
            else:
                if col == 'os':
                    select_posts_use['os_iOS'] = (select_posts_use[col] == 'iOS').astype(int)
                elif col == 'source':
                    select_posts_use['source_organic'] = (select_posts_use[col] == 'organic').astype(int)
                select_posts_use = select_posts_use.drop(col, axis=1)
        else:
            means = select_posts_use.groupby(col)['target'].mean()
            noisy_means = means + C * np.random.randn(len(means))
            mapping_dict = noisy_means.to_dict()
            select_posts_use[col] = select_posts_use[col].map(mapping_dict)
    
    select_posts_use = select_posts_use.drop(['target'], axis = 1) 
    select_posts_use['day'] = select_posts_use['min_date'].dt.dayofweek
    select_posts_use['month'] = select_posts_use['min_date'].dt.month
    select_posts_use['hour'] = select_posts_use['min_date'].dt.hour
    select_posts_use = select_posts_use.drop(['min_date'],axis=1)

    select_posts_use['predict'] = model_control.predict_proba(select_posts_use)[:,1]
    typle_idx = select_posts_use['predict'].nlargest(limit).index.tolist()
    posts_idx = []
    for item in typle_idx:
        posts_idx.append(item[1])
    #Вывести мы должны список из словарей
    final_posts = []
    for idx in posts_idx:
        myDict = {
            'id':  idx,
            'text': posts.loc[idx, 'text'],
            'topic': posts.loc[idx, 'topic'],
        }
        final_posts.append(myDict)
    return final_posts
      
#Функция DL
def recommendations_DL(id: int, time: datetime, limit: int):
    user_view_posts = (features_data[features_data['user_id'] == id]
    [['post_id', 'timestamp']].drop_duplicates(subset=['post_id']).reset_index(drop=True)
                      )
    viewed_ids = user_view_posts['post_id'].tolist() 
    select_posts = preprocess_posts_new[~preprocess_posts_new['post_id'].isin(viewed_ids)]
    user_init = user_data[user_data['user_id'] == id]
    select_posts_use = select_posts.copy()
    user_dates = features_data[features_data['user_id'] == id]['timestamp']
    user_dates_mean = user_dates.mean()
    select_posts_use.insert(0, 'month', user_dates_mean.month)
    select_posts_use.insert(1, 'hour', user_dates_mean.hour)
    for col in user_init.columns:
        select_posts_use = select_posts_use.assign(**{col: user_init[col].iloc[0]})
    new_order = list(user_init.columns) + [col for col in select_posts_use.columns if col not in user_init.columns]
    select_posts_use = select_posts_use[new_order]
    select_posts_use = select_posts_use.set_index(['user_id', 'post_id'])
    select_posts_use.drop(['index'], axis = 1, inplace=True)
    select_posts_use['predict'] = model_test.predict_proba(select_posts_use)[:,1]
    typle_idx = select_posts_use['predict'].nlargest(limit).index.tolist()
    posts_idx = []
    for item in typle_idx:
        posts_idx.append(item[1])
    final_posts = []
    for idx in posts_idx:
        myDict = {
            'id':  idx,
            'text': posts.loc[idx, 'text'],
            'topic': posts.loc[idx, 'topic'],
        }
        final_posts.append(myDict)
    return final_posts

@app.get("/post/recommendations/", response_model=Response)
def get_recommended_feed(id: int, time: datetime, limit: int, exp_group: str = None) -> Response:
    try:
        # Валидация входных данных
        if time < datetime(2021, 10, 1):
            raise ValueError("Дата должна быть позже 01.10.2021")
        if limit > 20:
            raise ValueError("Лимит не может превышать 20 постов")
        if id < 200 or id > 168552:
            raise ValueError("Пользователь не найден, невозможно подобрать рекомендацию")
        
        if exp_group is None:
            exp_group = get_exp_group(id)
        if exp_group == 'test':
            logger.info(f'Dl model for user {id}')
            recs = recommendations_DL(id, time, limit)
        elif exp_group == 'control':
            logger.info(f'ML model for user {id}')
            recs = recommendations_ML(id, time, limit)
        else:
            raise HTTPException(status_code=404, detail="No recommendations found")
    
        return Response(exp_group=exp_group, recommendations=recs)
    
    except ValueError as e:
        # Логируем ошибку валидации
        logger.warning(f"Validation error for user {id}: {str(e)}")
        # Показываем пользователю понятное сообщение
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Логируем непредвиденные ошибки
        logger.error(f"Error processing user {id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    