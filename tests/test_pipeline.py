"""Checks which detect leakage, schema breaks, persistence and app failures."""
import importlib.util
import json
import os
import unittest
import joblib
import numpy as np
import pandas as pd
from tourism_project.model_building.common import (DATA,ROOT,SPLITS,DEPLOY,FEATURES,POST_CONTACT,
    clean_data,profile_groups)
from tourism_project.model_building.data_register import validate

class PipelineTests(unittest.TestCase):
    def test_registration_rejects_missing_target(self):
        with self.assertRaises(ValueError): validate(pd.read_csv(DATA).drop(columns='ProdTaken'))

    def test_registration_rejects_bad_target(self):
        d=pd.read_csv(DATA);d.loc[0,'ProdTaken']=2
        with self.assertRaises(ValueError):validate(d)

    def test_no_post_contact_or_identifier_features(self):
        x=pd.read_csv(SPLITS/'X_train.csv')
        self.assertEqual(list(x),FEATURES)
        self.assertFalse(set(x)&set(POST_CONTACT+['CustomerID','ProdTaken','Unnamed: 0']))

    def test_no_matching_profiles_across_partitions(self):
        a=pd.read_csv(SPLITS/'X_train.csv');b=pd.read_csv(SPLITS/'X_test.csv')
        self.assertFalse(set(profile_groups(a))&set(profile_groups(b)))

    def test_cleaning_is_idempotent_on_predictors(self):
        d=clean_data(pd.read_csv(DATA))
        self.assertFalse(d.duplicated().any())
        self.assertNotIn('Fe Male',set(d.Gender))

    def test_saved_model_reproduces_evaluation(self):
        model=joblib.load(DEPLOY/'model.joblib');x=pd.read_csv(SPLITS/'X_test.csv')
        expected=pd.read_csv(ROOT/'reports/test_predictions.csv').purchase_score
        np.testing.assert_allclose(model['pipeline'].predict_proba(x)[:,1],expected,atol=1e-12)

    def test_unseen_categories_and_missing_values(self):
        model=joblib.load(DEPLOY/'model.joblib');x=pd.read_csv(SPLITS/'X_test.csv').head(1).copy()
        x['Occupation']='Previously unseen occupation';x['Age']=np.nan
        prob=model['pipeline'].predict_proba(x)
        self.assertTrue(np.isfinite(prob).all());self.assertAlmostEqual(float(prob.sum()),1.)

    def test_quality_gate(self):
        m=json.loads((DEPLOY/'metadata.json').read_text())
        self.assertGreater(m['test_metrics']['average_precision'],m['baseline_metrics']['average_precision'])
        self.assertGreaterEqual(m['test_metrics']['recall'],0.5)

    def test_streamlit_form(self):
        if importlib.util.find_spec('streamlit') is None:
            if os.getenv('REQUIRE_STREAMLIT')=='1':self.fail('Streamlit required in CI')
            self.skipTest('Streamlit is not installed locally; the workflow requires this test.')
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(DEPLOY/'app.py')).run(timeout=40)
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run(timeout=40)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.metric),2)

if __name__=='__main__':unittest.main(verbosity=2)
