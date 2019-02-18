from . import anchor_base
from . import anchor_explanation
from . import utils
import lime
import lime.lime_tabular
import collections
import sklearn
import numpy as np
import os
import copy
import string
from io import open
import json

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

class AnchorTabularExplainer(object):
    """
    Args:
        class_names: list of strings
        feature_names: list of strings
        data: used to fit feature pre-processor
        categorical_names: map from integer to list of strings, names for each
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal, and thus discretized.
        encoder: feature preprocessing function with 'fit' and 'transform' methods
    """
    def __init__(self, class_names, feature_names, data=None, 
                 categorical_names=None, encoder=None):
        
        if encoder: # assign and fit feature preprocessing function
            self.encoder = encoder
            self.encoder.fit(data)
        else:
            self.encoder = collections.namedtuple('encoder',['transform'])(lambda x: x)
        
        self.categorical_features = sorted(categorical_names.keys())
        self.ordinal_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]
        
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names # dict with {col: categorical feature options}
        
    
    def fit(self, train_data, train_labels, validation_data, 
            validation_labels, discretizer='quartile'):
        """
        Apply discretization to ordinal features.
        """
        self.train = train_data
        self.train_labels = train_labels
        self.validation = validation_data
        self.validation_labels = validation_labels
        
        # quartile or decile discretization of ordinal features
        args = [train_data, self.categorical_features, self.feature_names]
        if discretizer == 'quartile':
            self.disc = lime.lime_tabular.QuartileDiscretizer(*args)
        elif discretizer == 'decile':
            self.disc = lime.lime_tabular.DecileDiscretizer(*args)
        else:
            raise ValueError('Discretizer must be quartile or decile')
        
        self.d_train = self.disc.discretize(self.train)
        self.d_validation = self.disc.discretize(self.validation)
        
        # add discretized ordinal features to categorical features
        self.categorical_names.update(self.disc.names)
        self.categorical_features += self.ordinal_features
        
        # calculate min, max and std for ordinal features in training data
        self.min = {}
        self.max = {}
        self.std = {}
        for f in range(train_data.shape[1]):
            if f in self.categorical_features and f not in self.ordinal_features:
                continue
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])
            self.std[f] = np.std(train_data[:, f])

    
    def sample_from_train(self, conditions_eq, conditions_neq, conditions_geq,
                          conditions_leq, num_samples, validation=True):
        """
        - sample data with feature values in same discretized bin / same categorical values as the observation to be explained
        - "sample from validation" if the feature is in the anchor
        """
        # set train set to validation set if validation=True -> basically sample from validation set
        train = self.train if not validation else self.validation # original data
        d_train = self.d_train if not validation else self.d_validation # binned data
        
        # sample from train and d_train (which can both be validation...) set with replacement
        idx = np.random.choice(range(train.shape[0]), num_samples,replace=True)
        sample = train[idx]
        d_sample = d_train[idx]
        
        # for each sampled data point, fill in the categorical feature values with the value ...
        # ... from the same categorical feature in the data to be explained
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)
        
        # value geq: largest bin value smaller than the observation to be explained value for the ordinal feature
        for f in conditions_geq:
            
            # idx of samples where feature value is in a lower bin than the observation to be explained
            idx = d_sample[:, f] <= conditions_geq[f]
            
            # add idx where feature value is in a higher bin than the observation
            if f in conditions_leq:
                idx = (idx + (d_sample[:, f] > conditions_leq[f])).astype(bool)
                
            if idx.sum() == 0: # if all values in sampled data have same bin as explained obs -> continue 
                continue
            
            # options = idx in train (not sample) dataset where feature value is in same bin than observation to be explained
            options = d_train[:, f] > conditions_geq[f]
            if f in conditions_leq:
                options = options * (d_train[:, f] <= conditions_leq[f])
            
            # draw random samples from training set ...
            if options.sum() == 0: # ... uniformly sampled between min and max of feature if no 
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else: # ... with options=True
                to_rep = np.random.choice(train[options, f], idx.sum(), replace=True)
            
            # replace sample values for ordinal features where feature values are in a lower/higher bin than the observation ...
            # ... by random values from training set from same bin
            sample[idx, f] = to_rep
        
        # apply same principle (replacing values outside of observation to be explained bin by random value from ...
        # ... training set from same bin) for values in leq
        for f in conditions_leq:
            if f in conditions_geq:
                continue
            idx = d_sample[:, f] > conditions_leq[f]
            if idx.sum() == 0:
                continue
            options = d_train[:, f] <= conditions_leq[f]
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(), replace=True)
            sample[idx, f] = to_rep
        
        return sample
    

    def transform_to_examples(self, examples, features_in_anchor=[],
                              predicted_label=None):
        ret_obj = []
        if len(examples) == 0:
            return ret_obj
        weights = [int(predicted_label) if x in features_in_anchor else -1
                   for x in range(examples.shape[1])]
        examples = self.disc.discretize(examples)
        for ex in examples:
            values = [self.categorical_names[i][int(ex[i])]
                      if i in self.categorical_features
                      else ex[i] for i in range(ex.shape[0])]
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        return ret_obj

    def to_explanation_map(self, exp):
        def jsonize(x): return json.dumps(x)
        instance = exp['instance']
        predicted_label = exp['prediction']
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            features_in_anchor = set(exp['feature'][:i])
            ret = {}
            ret['coveredFalse'] = self.transform_to_examples(
                temp['covered_false'], features_in_anchor, predicted_label)
            ret['coveredTrue'] = self.transform_to_examples(
                temp['covered_true'], features_in_anchor, predicted_label)
            ret['uncoveredTrue'] = self.transform_to_examples(
                temp['uncovered_true'], features_in_anchor, predicted_label)
            ret['uncoveredFalse'] = self.transform_to_examples(
                temp['uncovered_false'], features_in_anchor, predicted_label)
            ret['covered'] =self.transform_to_examples(
                temp['covered'], features_in_anchor, predicted_label)
            examples_obj.append(ret)

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': examples_obj,
                       'onlyShowActive': False}
        weights = [-1 for x in range(instance.shape[0])]
        instance = self.disc.discretize(exp['instance'].reshape(1, -1))[0]
        values = [self.categorical_names[i][int(instance[i])]
                  if i in self.categorical_features
                  else instance[i] for i in range(instance.shape[0])]
        raw_data = list(zip(self.feature_names, values, weights))
        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }
        return ret

    def as_html(self, exp, **kwargs):
        """bla"""
        exp_map = self.to_explanation_map(exp)

        def jsonize(x): return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(exp_map['labelNames']),
                            predict_proba=jsonize(exp_map['predictProba']),
                            true_class=jsonize(exp_map['trueClass']),
                            explanation=jsonize(exp_map['explanation']),
                            raw_data=jsonize(exp_map['rawData']),
                            explanation_type=jsonize(exp_map['explanationType']))
        out += u'</body></html>'
        return out

    
    def get_sample_fn(self, data_row, classifier_fn, desired_label=None):
        """
        - get true label for observation to be explained
        - create mapping: dict with 
            key = idx
            value = (feature, 'eq'/'leq'/'geq', value of feature)
        - create sample_fn function
        
        Returns:
            sample_fn : sampling function which returns raw sampled data, categorical sample data and labels
            mapping : maps features to categories or bins (for ordinal features which are discretized)
        """
        # predict function of classifier on pre-processed data
        def predict_fn(x):
            return classifier_fn(self.encoder.transform(x))
        
        # if no true label available; true label = predicted label
        true_label = desired_label
        if true_label is None:
            true_label = predict_fn(data_row.reshape(1, -1))[0]
        
        # discretize ordinal features of data to be explained
        # create mapping dict:
        # mapping = (feature id, 'eq' (cat) or 'leq' and 'geq' (ordinal bins lower or greater than feature value), feature value)
        mapping = {}
        data_row = self.disc.discretize(data_row.reshape(1, -1))[0] # bin ordinal features in obs to be explained
        for f in self.categorical_features:
            if f in self.ordinal_features:
                for v in range(len(self.categorical_names[f])): # loop over nb of bins for the binned ordinal features
                    idx = len(mapping)
                    # data value lower than bin value - store bin value
                    if data_row[f] <= v and v != len(self.categorical_names[f]) - 1:
                        mapping[idx] = (f, 'leq', v)
                        # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                    # data value higher than bin value - store bin value
                    elif data_row[f] > v:
                        mapping[idx] = (f, 'geq', v)
                        # names[idx] = '%s > %s' % (self.feature_names[f], v)
            else:
                idx = len(mapping)
                mapping[idx] = (f, 'eq', data_row[f])
            # names[idx] = '%s = %s' % (
            #     self.feature_names[f],
            #     self.categorical_names[f][int(data_row[f])])

        def sample_fn(present, num_samples, compute_labels=True, validation=True):
            """
            - present : list with keys of mapping dict to (feature, 'eq'/'leq'/'geq', value of feature)
            
            Returns:
                raw_data : sampled data with feature values in same discretized bin / same categorical values as
                           observation to be explained
                data : raw_data in categorical form using mapping -> adding extra columns for 'leq'/'geq'
                labels : whether the predictions on the "perturbed" data is the same as the (proxy, through prediction)
                         label of the data to be explained
            """
            # 3 dicts, fill with 'eq'/'leq'/'geq' from mapping
            # key = feature idx
            # value = feature value
            conditions_eq = {}
            conditions_leq = {}
            conditions_geq = {}
            for x in present: # x is a feature (binned for ordinal features) and a key from the mapping dict
                f, op, v = mapping[x] # (feature, 'eq'/'leq'/'geq', feature value)
                if op == 'eq':
                    conditions_eq[f] = v
                if op == 'leq':
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    # only store smallest bin larger than feature value for ordinal feature
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == 'geq':
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    # only store largest bin smaller than feature value for ordinal feature
                    conditions_geq[f] = max(conditions_geq[f], v)
            # conditions_eq = dict([(x, data_row[x]) for x in present])
            
            # sample data with feature values in same discretized bin / same categorical values as ...
            # ... the observation to be explained
            raw_data = self.sample_from_train(conditions_eq, {}, conditions_geq,
                                              conditions_leq, num_samples,validation=validation)
            
            # discretize sampled data
            d_raw_data = self.disc.discretize(raw_data)
            
            # use raw_data (sampled) to fill in data array with all categorical features ...
            # ... using the mapping
            # data = all bins for ordinal data + categorical data
            # 1 if in bin, 0 otherwise
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)
            # data = (raw_data == data_row).astype(int)
            
            # create labels using model predictions as true labels
            labels = []
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels
        return sample_fn, mapping
    
    
    def explain_instance(self, data_row, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100,
                          max_anchor_size=None,
                          desired_label=None,
                          beam_size=4, **kwargs):
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(
            data_row, classifier_fn, desired_label=desired_label)
        # return sample_fn, mapping
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, max_anchor_size=max_anchor_size,
            **kwargs)
        self.add_names_to_exp(data_row, exp, mapping)
        exp['instance'] = data_row
        exp['prediction'] = classifier_fn(self.encoder.transform(data_row.reshape(1, -1)))[0]
        explanation = anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        return explanation

    def add_names_to_exp(self, data_row, hoeffding_exp, mapping):
        # TODO: precision recall is all wrong, coverage functions wont work
        # anymore due to ranges
        idxs = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]
        ordinal_ranges = {}
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)
        handled = set()
        for idx in idxs:
            f, op, v = mapping[idx]
            # v = data_row[f]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                            or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.categorical_names[f]) - 1:
                        geq = geq - 1
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                               leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            hoeffding_exp['names'].append(fname)
