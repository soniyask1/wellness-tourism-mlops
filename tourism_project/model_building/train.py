"""Group-aware model search, out-of-fold threshold selection, and one holdout evaluation.

CI requires MLflow. A clearly labelled portable JSON tracker lets the same ML
experiment execute in offline notebook environments where MLflow is unavailable.
"""
import argparse
import json
import os
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, fbeta_score,
    f1_score, average_precision_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, brier_score_loss)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tourism_project.model_building.common import (ROOT, DATA, SPLITS, DEPLOY,
    REPORTS, NUMERIC, CATEGORICAL, FEATURES, dump_json, sha256, profile_groups)

def preprocessor():
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)])

def score(y, prob, threshold):
    pred = np.asarray(prob) >= threshold
    return {'accuracy': accuracy_score(y,pred), 'precision': precision_score(y,pred,zero_division=0),
            'recall': recall_score(y,pred,zero_division=0), 'f1': f1_score(y,pred,zero_division=0),
            'f2': fbeta_score(y,pred,beta=2,zero_division=0), 'roc_auc': roc_auc_score(y,prob),
            'average_precision': average_precision_score(y,prob),
            'brier_score': brier_score_loss(y,prob), 'contact_fraction': float(pred.mean())}

def run(require_mlflow=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
    start = time.perf_counter(); REPORTS.mkdir(exist_ok=True); DEPLOY.mkdir(exist_ok=True)
    xtrain = pd.read_csv(SPLITS/'X_train.csv'); xtest = pd.read_csv(SPLITS/'X_test.csv')
    ytrain = pd.read_csv(SPLITS/'y_train.csv').iloc[:,0]; ytest = pd.read_csv(SPLITS/'y_test.csv').iloc[:,0]
    groups = pd.read_csv(SPLITS/'groups_train.csv',dtype=str).iloc[:,0]
    gtest = pd.read_csv(SPLITS/'groups_test.csv',dtype=str).iloc[:,0]
    assert list(xtrain) == FEATURES == list(xtest)
    assert not set(groups) & set(gtest)
    assert groups.tolist() == profile_groups(xtrain).tolist(), 'Group manifest does not match train data'
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=17)
    fold_indices = list(cv.split(xtrain, ytrain, groups))
    for a,b in fold_indices:
        assert not set(groups.iloc[a]) & set(groups.iloc[b])
    pipeline = Pipeline([('preprocess', preprocessor()), ('model', RandomForestClassifier())])
    # 16 candidates x 3 folds = 48 fits. The test partition is never supplied to this search.
    grid = [
        {'model': [RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=1,class_weight='balanced')],
         'model__max_depth': [None, 10], 'model__min_samples_leaf': [1, 3], 'model__max_features': ['sqrt', 0.7]},
        {'model': [GradientBoostingClassifier(random_state=42)], 'model__n_estimators': [100, 180],
         'model__learning_rate': [0.05, 0.1], 'model__max_depth': [2, 3]}]
    search = GridSearchCV(pipeline, grid, scoring={'ap':'average_precision','recall':'recall','roc_auc':'roc_auc'},
                          refit='ap', cv=fold_indices, n_jobs=2, error_score='raise', return_train_score=True)
    print('Searching 16 candidates with 3 group-disjoint folds; ranking by validation average precision...', flush=True)
    search.fit(xtrain,ytrain); best = search.best_estimator_
    results = pd.DataFrame(search.cv_results_)
    # Save every candidate, every varied/fixed parameter, and every fold score.
    serial = results.copy()
    serial['params'] = serial.params.map(lambda d: json.dumps({k:str(v) for k,v in d.items()},sort_keys=True))
    serial.to_csv(REPORTS/'cv_results.csv',index=False)
    experiment_id = time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    tracker = None; mlflow_error = None
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///'+str(ROOT/'artifacts/mlflow.db')))
        mlflow.set_experiment('wellness-precontact-purchase')
        tracker = mlflow
    except ImportError as exc:
        if require_mlflow: raise RuntimeError('MLflow is required by CI; install requirements-ci.txt') from exc
        mlflow_error = 'MLflow not installed in this execution environment; portable JSON tracking was used.'
        print(mlflow_error)
    trials = []
    parent = tracker.start_run(run_name=experiment_id) if tracker else None
    try:
        if tracker:
            tracker.log_params({'dataset_sha256':sha256(DATA),'split_seed':42,'cv_seed':17,
                                'cv_folds':3,'selection_metric':'average_precision','features':','.join(FEATURES)})
        for i,row in results.iterrows():
            # Include the estimator's fixed defaults, not just values varied in the grid.
            model = row['params']['model']; params = model.get_params().copy()
            params.update({k.removeprefix('model__'):v for k,v in row['params'].items() if k!='model'})
            params = {k:str(v) for k,v in params.items()}; params['algorithm']=type(model).__name__
            metrics = {c:float(row[c]) for c in results if c.startswith(('mean_test_','std_test_','split')) and '_test_' in c}
            trials.append({'trial':int(i),'parameters':params,'metrics':metrics,'rank_ap':int(row['rank_test_ap'])})
            if tracker:
                with tracker.start_run(run_name=f'candidate-{i:02}',nested=True):
                    tracker.log_params(params); tracker.log_metrics(metrics)
        # Only training-fold predictions choose the decision threshold.
        oof = cross_val_predict(best, xtrain,ytrain,cv=fold_indices,method='predict_proba',n_jobs=2)[:,1]
        precision, recall, thresholds = precision_recall_curve(ytrain,oof)
        f2 = 5*precision[:-1]*recall[:-1] / np.maximum(4*precision[:-1]+recall[:-1],1e-12)
        threshold = float(thresholds[np.argmax(f2)])
        prob = best.predict_proba(xtest)[:,1]; pred = (prob >= threshold).astype(int)
        metrics = score(ytest,prob,threshold)
        baseline = DummyClassifier(strategy='prior').fit(xtrain,ytrain)
        base_prob = baseline.predict_proba(xtest)[:,1]
        baseline_metrics = score(ytest,base_prob,0.5)
        selection = {'algorithm':type(best.named_steps['model']).__name__,
                     'best_parameters':{k:str(v) for k,v in search.best_params_.items()},
                     'best_cv_average_precision':float(search.best_score_), 'threshold':threshold,
                     'threshold_rule':'maximize F2 using training out-of-fold probabilities',
                     'training_oof_f2':float(f2.max()), 'test_metrics':metrics,
                     'test_metrics_threshold_05':score(ytest,prob,0.5), 'baseline_metrics':baseline_metrics,
                     'confusion_matrix':confusion_matrix(ytest,pred,labels=[0,1]).tolist(),
                     'test_rows':len(ytest), 'train_rows':len(ytrain), 'dataset_sha256':sha256(DATA),
                     'python_version':platform.python_version(),'sklearn_version':sklearn.__version__,
                     'features':FEATURES, 'tracking_backend':'MLflow + portable JSON' if tracker else 'portable JSON only',
                     'mlflow_status':mlflow_error or 'executed', 'experiment_id':experiment_id,
                     'holdout_policy':'held-out test evaluated after model and threshold selection; not used for tuning'}
        # Bootstrap quantifies test-sample variability; it is not a causal/campaign estimate.
        rng = np.random.default_rng(19); boot=[]
        # Resample entire profile groups because rows in one profile are not independent.
        cluster_rows = [np.flatnonzero(gtest.to_numpy()==g) for g in gtest.unique()]
        for _ in range(500):
            selected=rng.integers(0,len(cluster_rows),len(cluster_rows))
            idx=np.concatenate([cluster_rows[i] for i in selected]); yy=ytest.to_numpy()[idx]
            if len(np.unique(yy))==2: boot.append([average_precision_score(yy,prob[idx]),recall_score(yy,pred[idx],zero_division=0)])
        selection['bootstrap_95_ci']={k:np.quantile(np.array(boot)[:,i],[.025,.975]).tolist() for i,k in enumerate(['average_precision','recall'])}
        selection['bootstrap_method']='500 profile-cluster bootstrap resamples; descriptive test uncertainty'
        dump_json(REPORTS/'experiment_trials.json',trials); dump_json(REPORTS/'metrics.json',selection)
        dump_json(DEPLOY/'metadata.json',selection)
        joblib.dump({'pipeline':best, 'threshold':threshold,'features':FEATURES},DEPLOY/'model.joblib',compress=3)
        pd.DataFrame({'actual':ytest,'purchase_score':prob,'prediction':pred}).to_csv(REPORTS/'test_predictions.csv',index=False)
        print('\nSelected model:',selection['algorithm']); print('Best CV average precision:',round(search.best_score_,4))
        print('Training-selected threshold:',round(threshold,4)); print(pd.Series(metrics).round(4).to_string())
        print('\n',classification_report(ytest,pred,digits=3,zero_division=0))
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        ConfusionMatrixDisplay.from_predictions(ytest,pred,ax=axes[0],colorbar=False,cmap='Blues')
        PrecisionRecallDisplay.from_predictions(ytest,prob,ax=axes[1]); axes[1].axhline(ytest.mean(),ls='--',color='gray',label='Test prevalence'); axes[1].legend(fontsize=8)
        RocCurveDisplay.from_predictions(ytest,prob,ax=axes[2]); axes[2].plot([0,1],[0,1],'--',color='gray')
        fig.suptitle('Untouched holdout • pre-contact features only',fontsize=15); fig.tight_layout()
        fig.savefig(REPORTS/'evaluation.png',dpi=160,bbox_inches='tight'); plt.close(fig)
        # Held-out permutation importance is diagnostic only, after all decisions are fixed.
        from sklearn.inspection import permutation_importance
        importance=permutation_importance(best,xtest,ytest,n_repeats=5,random_state=42,scoring='average_precision',n_jobs=2)
        imp=pd.DataFrame({'feature':FEATURES,'mean_ap_drop':importance.importances_mean,'std':importance.importances_std}).sort_values('mean_ap_drop',ascending=False)
        imp.to_csv(REPORTS/'feature_importance.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,5)); p=imp.sort_values('mean_ap_drop'); ax.barh(p.feature,p.mean_ap_drop,xerr=p['std'],color='#167b83'); ax.set_xlabel('Average precision decrease after shuffling'); ax.set_title('Holdout permutation importance • association, not causation'); fig.tight_layout(); fig.savefig(REPORTS/'importance.png',dpi=150); plt.close(fig)
        # Exploratory gender slice metrics to flag disparity; small groups are unstable.
        slices=[]
        for value in sorted(xtest.Gender.unique()):
            mask=xtest.Gender==value
            slices.append({'gender':value,'n':int(mask.sum()),'buyers':int(ytest[mask].sum()),**score(ytest[mask],prob[mask],threshold)})
        pd.DataFrame(slices).to_csv(REPORTS/'gender_audit.csv',index=False)
        if tracker:
            tracker.log_params({'algorithm':selection['algorithm'],'threshold':threshold})
            tracker.log_metrics({f'test_{k}':v for k,v in metrics.items()})
            tracker.log_metric('best_cv_average_precision',float(search.best_score_))
            for p in [REPORTS/'cv_results.csv',REPORTS/'metrics.json',REPORTS/'experiment_trials.json',REPORTS/'evaluation.png',DEPLOY/'model.joblib']:
                tracker.log_artifact(str(p))
        print('Tracking backend:',selection['tracking_backend']); print(f'Elapsed: {time.perf_counter()-start:.1f}s')
    finally:
        if tracker: tracker.end_run()
    return selection

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--require-mlflow',action='store_true'); run(p.parse_args().require_mlflow)
