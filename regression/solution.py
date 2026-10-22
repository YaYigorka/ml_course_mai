# Лабораторная работа №1. Регрессия

# Предсказание качества вина (UCI Wine Quality)

# --- Исходная code cell 1 ---
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (10, 5)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# 1. Загрузка данных

# --- Исходная code cell 3 ---
BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'

def load_wine(kind):
    fname = f'winequality-{kind}.csv'
    try:
        return pd.read_csv(fname, sep=';')            # локальный файл
    except FileNotFoundError:
        return pd.read_csv(BASE_URL + fname, sep=';')  # загрузка с UCI

red = load_wine('red')
white = load_wine('white')
red['is_red'] = 1
white['is_red'] = 0

df = pd.concat([red, white], ignore_index=True)
# приводим названия к удобному виду: fixed acidity -> fixed_acidity
df.columns = [c.replace(' ', '_') for c in df.columns]

print(f'Красное: {red.shape},  белое: {white.shape},  всего: {df.shape}')
df.head()


# --- Исходная code cell 4 ---
df.info()


# 2. Разведочный анализ данных (EDA)

# 2.1. Описательные статистики и дубликаты

# --- Исходная code cell 7 ---
df.describe().T


# --- Исходная code cell 8 ---
n_dup = df.duplicated().sum()
print(f'Полных дубликатов строк: {n_dup} ({n_dup / len(df):.1%})')


# 2.2. Распределение целевой переменной

# --- Исходная code cell 11 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

counts = df['quality'].value_counts().sort_index()
axes[0].bar(counts.index, counts.values, color='steelblue')
for q, c in counts.items():
    axes[0].text(q, c + 30, f'{c}\n({c / len(df):.1%})', ha='center', fontsize=9)
axes[0].set_xlabel('quality'); axes[0].set_ylabel('Количество'); axes[0].set_title('Распределение quality (все вина)')
axes[0].set_ylim(0, counts.max() * 1.2)

ct = pd.crosstab(df['quality'], df['is_red'].map({1: 'красное', 0: 'белое'}), normalize='columns')
ct.plot(kind='bar', ax=axes[1], color=['gold', 'darkred'], rot=0)
axes[1].set_title('Доля оценок внутри каждого типа вина'); axes[1].set_ylabel('Доля')
plt.tight_layout(); plt.show()

print('Среднее quality:', round(df['quality'].mean(), 3), ' медиана:', df['quality'].median(),
      ' std:', round(df['quality'].std(), 3))


# 2.3. Распределения признаков

# --- Исходная code cell 14 ---
features = [c for c in df.columns if c not in ('quality', 'is_red')]
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(data=df, x=col, hue=df['is_red'].map({1: 'красное', 0: 'белое'}),
                 bins=40, ax=ax, stat='density', common_norm=False, element='step',
                 palette={'белое': 'goldenrod', 'красное': 'darkred'})
    ax.set_title(col); ax.set_xlabel(''); ax.legend_.set_title('')
axes.ravel()[-1].axis('off')
plt.suptitle('Распределения признаков для красного и белого вина', y=1.01, fontsize=14)
plt.tight_layout(); plt.show()


# 2.4. Корреляции

# --- Исходная code cell 17 ---
corr = df.corr()
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={'width_ratios': [3, 1.3]})

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=axes[0], cbar_kws={'shrink': .8}, annot_kws={'size': 8})
axes[0].set_title('Корреляционная матрица (Пирсон)')

q_corr = corr['quality'].drop('quality').sort_values()
colors = ['indianred' if v < 0 else 'seagreen' for v in q_corr]
axes[1].barh(q_corr.index, q_corr.values, color=colors)
axes[1].set_title('Корреляция признаков с quality'); axes[1].axvline(0, color='k', lw=.8)
plt.tight_layout(); plt.show()

q_corr.sort_values(key=abs, ascending=False).to_frame('corr_with_quality')


# 2.5. Связь ключевых признаков с качеством

# --- Исходная code cell 20 ---
top_feats = ['alcohol', 'density', 'volatile_acidity', 'chlorides']
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, col in zip(axes, top_feats):
    sns.boxplot(data=df, x='quality', y=col, ax=ax, color='lightsteelblue', showfliers=False)
    ax.set_title(f'{col} по уровням quality')
plt.tight_layout(); plt.show()

df.groupby('quality')[top_feats].median()


# 3. Разбиение на обучающую и тестовую выборки

# --- Исходная code cell 23 ---
X = df.drop(columns='quality')
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print(f'train: {X_train.shape},  test: {X_test.shape}')
pd.DataFrame({
    'train, доля': y_train.value_counts(normalize=True).sort_index(),
    'test, доля':  y_test.value_counts(normalize=True).sort_index(),
    'train, шт':   y_train.value_counts().sort_index(),
    'test, шт':    y_test.value_counts().sort_index(),
})


# --- Исходная code cell 25 ---
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}  # сюда собираем итоговые метрики на тесте

def evaluate(name, model):
    # CV на train (для принятия решений) + финальная оценка на test
    cv_res = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                            scoring=('neg_root_mean_squared_error', 'neg_mean_absolute_error'))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        'CV RMSE (train)': -cv_res['test_neg_root_mean_squared_error'].mean(),
        'CV MAE (train)':  -cv_res['test_neg_mean_absolute_error'].mean(),
        'RMSE': rmse(y_test, pred),
        'MAE':  mean_absolute_error(y_test, pred),
    }
    print(f"{name:20s}  CV RMSE = {results[name]['CV RMSE (train)']:.4f} | "
          f"test RMSE = {results[name]['RMSE']:.4f}, test MAE = {results[name]['MAE']:.4f}")
    return model, pred


# 4. Метрики качества и baseline

# 4.1. Метрики

# 4.2. Baseline — константное предсказание

# --- Исходная code cell 27 ---
mean_c = y_train.mean()
median_c = y_train.median()
pred_mean = np.full(len(y_test), mean_c)
pred_median = np.full(len(y_test), median_c)

print(f'Среднее train = {mean_c:.4f}, медиана train = {median_c}')
print(f'Baseline (mean):   RMSE = {rmse(y_test, pred_mean):.4f}, MAE = {mean_absolute_error(y_test, pred_mean):.4f}')
print(f'Baseline (median): RMSE = {rmse(y_test, pred_median):.4f}, MAE = {mean_absolute_error(y_test, pred_median):.4f}')

results['Baseline (mean)'] = {
    'CV RMSE (train)': np.nan, 'CV MAE (train)': np.nan,
    'RMSE': rmse(y_test, pred_mean), 'MAE': mean_absolute_error(y_test, pred_mean)}


# 5. Модели

# 5.1. Линейная регрессия

# --- Исходная code cell 30 ---
lr = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
lr, pred_lr = evaluate('Linear Regression', lr)


# --- Исходная code cell 31 ---
coef = pd.Series(lr.named_steps['model'].coef_, index=X.columns).sort_values()
raw_lr = LinearRegression().fit(X_train, y_train)
coef_table = pd.DataFrame({
    'коэф. (стандартизованные)': coef,
    'коэф. (исходный масштаб)': pd.Series(raw_lr.coef_, index=X.columns)[coef.index],
    'corr с quality': corr['quality'][coef.index],
})

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(coef.index, coef.values, color=['indianred' if v < 0 else 'seagreen' for v in coef])
ax.axvline(0, color='k', lw=.8)
ax.set_title('Коэффициенты линейной регрессии (стандартизованные признаки)')
ax.set_xlabel('Изменение прогноза quality при +1 std признака')
plt.tight_layout(); plt.show()
print('Свободный член:', round(lr.named_steps['model'].intercept_, 4))
coef_table


# 5.2. Случайный лес (Random Forest)

# --- Исходная code cell 34 ---
rf_grid = GridSearchCV(
    RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    param_grid={'max_features': [0.33, 0.5, 1.0], 'min_samples_leaf': [1, 2, 4]},
    scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
rf_grid.fit(X_train, y_train)
print('Лучшие параметры:', rf_grid.best_params_, '| CV RMSE =', round(-rf_grid.best_score_, 4))

pd.DataFrame(rf_grid.cv_results_).pivot(index='param_min_samples_leaf', columns='param_max_features',
                                        values='mean_test_score').mul(-1).round(4)


# --- Исходная code cell 35 ---
rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, **rf_grid.best_params_)
rf, pred_rf = evaluate('Random Forest', rf)


# --- Исходная code cell 37 ---
def plot_importance(imp, title, ax, top=5):
    imp = imp.sort_values()
    colors = ['darkorange' if i >= len(imp) - top else 'lightgray' for i in range(len(imp))]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title(title)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f' {v:.3f}', va='center', fontsize=8)

imp_rf = pd.Series(rf.feature_importances_, index=X.columns)
fig, ax = plt.subplots(figsize=(9, 5.5))
plot_importance(imp_rf, 'Random Forest: feature_importances_ (MDI), топ-5 выделены', ax)
plt.tight_layout(); plt.show()


# 5.3. Градиентный бустинг (Gradient Boosting)

# --- Исходная code cell 40 ---
gb_default = GradientBoostingRegressor(random_state=RANDOM_STATE)
cv_default = -cross_val_score(gb_default, X_train, y_train, cv=cv,
                              scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
print(f'GB с параметрами по умолчанию (100 деревьев, depth=3, lr=0.1): CV RMSE = {cv_default:.4f}')


# --- Исходная code cell 41 ---
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE, subsample=0.8, learning_rate=0.05),
    param_grid={'n_estimators': [300, 600, 1000], 'max_depth': [4, 6, 8]},
    scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
gb_grid.fit(X_train, y_train)
print('Лучшие параметры:', gb_grid.best_params_, '| CV RMSE =', round(-gb_grid.best_score_, 4))

pd.DataFrame(gb_grid.cv_results_).pivot(index='param_max_depth', columns='param_n_estimators',
                                        values='mean_test_score').mul(-1).round(4)


# --- Исходная code cell 42 ---
gb = GradientBoostingRegressor(random_state=RANDOM_STATE, subsample=0.8, learning_rate=0.05,
                               **gb_grid.best_params_)
gb, pred_gb = evaluate('Gradient Boosting', gb)


# --- Исходная code cell 44 ---
imp_gb = pd.Series(gb.feature_importances_, index=X.columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
plot_importance(imp_rf, 'Random Forest (MDI)', axes[0])
plot_importance(imp_gb, 'Gradient Boosting (MDI)', axes[1])
plt.tight_layout(); plt.show()


# 6. Сравнение моделей

# --- Исходная code cell 47 ---
res = pd.DataFrame(results).T.loc[['Baseline (mean)', 'Linear Regression', 'Random Forest', 'Gradient Boosting']]
res['RMSE vs baseline'] = (res['RMSE'] / res.loc['Baseline (mean)', 'RMSE'] - 1).map('{:+.1%}'.format)
res['RMSE < 0.70'] = res['RMSE'] < 0.70
res.index.name = 'Модель'
res


# --- Исходная code cell 48 ---
final_table = res[['RMSE', 'MAE']].copy()
display(final_table.round(4))

fig, ax = plt.subplots(figsize=(10, 4.5))
final_table.plot(kind='bar', ax=ax, rot=0, color=['steelblue', 'darkorange'])
ax.axhline(0.70, color='red', ls='--', label='порог RMSE = 0.70')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9)
ax.set_title('Сравнение моделей на тестовой выборке'); ax.legend(); ax.set_ylabel('Ошибка, баллы')
plt.tight_layout(); plt.show()


# --- Исходная code cell 50 ---
train_full = X_train.assign(quality=y_train)
train_dedup = train_full.drop_duplicates()
Xd, yd = train_dedup.drop(columns='quality'), train_dedup['quality']
print(f'train без дубликатов: {len(train_dedup)} из {len(train_full)} строк')

dedup_scores = {}
for name, model in [('Linear Regression', lr), ('Random Forest', rf), ('Gradient Boosting', gb)]:
    dedup_scores[name] = -cross_val_score(model, Xd, yd, cv=cv,
                                          scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
dedup_scores['Baseline (mean)'] = yd.std(ddof=0)
pd.DataFrame({'CV RMSE (с дубликатами)': res['CV RMSE (train)'],
              'CV RMSE (без дубликатов)': pd.Series(dedup_scores)}).loc[res.index]


# 7. Важность признаков: сравнение моделей

# --- Исходная code cell 53 ---
perm = {}
for name, model in [('Linear Regression', lr), ('Random Forest', rf), ('Gradient Boosting', gb)]:
    r = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=RANDOM_STATE,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    perm[name] = pd.Series(r.importances_mean, index=X.columns)

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, (name, imp) in zip(axes, perm.items()):
    plot_importance(imp, f'{name}: permutation importance (рост RMSE)', ax)
plt.tight_layout(); plt.show()


# --- Исходная code cell 54 ---
lr_abs_coef = coef.abs()
rank = pd.DataFrame({
    '|corr| с quality (EDA)': corr['quality'].drop('quality').abs(),
    '|коэф.| LR': lr_abs_coef,
    'RF MDI': imp_rf, 'GB MDI': imp_gb,
    'RF perm': perm['Random Forest'], 'GB perm': perm['Gradient Boosting'], 'LR perm': perm['Linear Regression'],
}).rank(ascending=False).astype(int)
rank = rank.sort_values('RF MDI')
print('Ранги признаков (1 — самый важный):')
rank.style.background_gradient(cmap='YlOrRd_r', axis=None)


# 8. Анализ ошибок лучшей модели

# --- Исходная code cell 57 ---
cv_rank = res.drop('Baseline (mean)')['CV RMSE (train)'].sort_values()
best_name = cv_rank.index[0]
best_pred = {'Linear Regression': pred_lr, 'Random Forest': pred_rf, 'Gradient Boosting': pred_gb}[best_name]
print('CV RMSE моделей:\n', cv_rank.round(4).to_string())
print(f'\nЛучшая модель по CV: {best_name}; test RMSE = {rmse(y_test, best_pred):.4f}, '
      f'MAE = {mean_absolute_error(y_test, best_pred):.4f}')

err = X_test.copy()
err['quality'] = y_test
err['pred'] = best_pred
err['residual'] = err['quality'] - err['pred']      # > 0: модель занизила оценку
err['abs_err'] = err['residual'].abs()


# 8.1. Распределение остатков и «истина vs прогноз»

# --- Исходная code cell 59 ---
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

sns.histplot(err['residual'], bins=40, kde=True, ax=axes[0], color='steelblue')
axes[0].axvline(0, color='k', lw=1)
axes[0].set_title('Распределение остатков (y − ŷ)'); axes[0].set_xlabel('остаток')

jit = np.random.uniform(-0.15, 0.15, len(err))
axes[1].scatter(err['quality'] + jit, err['pred'], alpha=0.3, s=12)
axes[1].plot([3, 9], [3, 9], 'r--', label='идеальный прогноз')
sns.pointplot(x=err['quality'], y=err['pred'], ax=axes[1], native_scale=True, color='black',
              errorbar=None, markers='D', label='средний прогноз')
axes[1].set_xlabel('истинное quality (с джиттером)'); axes[1].set_ylabel('прогноз')
axes[1].set_title('Истинные vs предсказанные значения'); axes[1].legend()

axes[2].scatter(err['pred'], err['residual'], alpha=0.3, s=12)
axes[2].axhline(0, color='k', lw=1)
axes[2].set_xlabel('прогноз'); axes[2].set_ylabel('остаток'); axes[2].set_title('Остатки vs прогноз')
plt.tight_layout(); plt.show()

print(f"Средний остаток: {err['residual'].mean():+.4f}")
print(f"Доля прогнозов с ошибкой < 0.5: {(err['abs_err'] < 0.5).mean():.1%}, "
      f"< 1.0: {(err['abs_err'] < 1).mean():.1%}, > 1.5: {(err['abs_err'] > 1.5).mean():.1%}")


# 8.2. Качество по значениям quality

# --- Исходная code cell 62 ---
by_q = err.groupby('quality').agg(
    n=('residual', 'size'),
    mean_pred=('pred', 'mean'),
    bias=('residual', 'mean'),
    MAE=('abs_err', 'mean'),
    RMSE=('residual', lambda r: np.sqrt((r ** 2).mean())))
by_q['доля в train'] = y_train.value_counts(normalize=True).sort_index()

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.bar(by_q.index, by_q['RMSE'], color='indianred', alpha=.8, label='RMSE')
ax1.set_xlabel('истинное quality'); ax1.set_ylabel('RMSE на тесте')
ax2 = ax1.twinx()
ax2.plot(by_q.index, by_q['доля в train'], 'o-', color='steelblue', label='доля в train')
ax2.set_ylabel('доля в train'); ax2.grid(False)
ax1.set_title(f'{best_name}: ошибка по значениям quality vs частота класса')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.88)); plt.tight_layout(); plt.show()
by_q


# 8.3. Крупнейшие ошибки

# --- Исходная code cell 65 ---
show_cols = ['quality', 'pred', 'residual', 'is_red', 'alcohol', 'volatile_acidity', 'density',
             'chlorides', 'free_sulfur_dioxide', 'sulphates', 'residual_sugar']
top_err = err.sort_values('abs_err', ascending=False).head(10)[show_cols]
top_err


# --- Исходная code cell 66 ---
# сравним признаки крупнейших ошибок с типичными значениями для их истинной оценки и прогноза
typical = df.groupby('quality')[show_cols[4:]].median()
print('Медианы признаков по оценкам качества (по всему датасету):')
typical


# 9. Итоговые выводы

# --- Исходная code cell 69 ---
final_table.round(4)

