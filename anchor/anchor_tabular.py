from . import anchor_base
from . import anchor_explanation
import lime
import numpy as np


class AnchorTabularExplainer(object):

    def __init__(self, predict_fn, train_data, class_names, feature_names,
                 categorical_names=None, discretizer='quartile'):
        """
        Initialize the anchor tabular explainer.

        Parameters
        ----------
        predict_fn
            Model prediction function
        train_data
            Representative sample from the training data
        class_names
            List with target labels names
        feature_names
            List with feature names
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature
        discretizer
            Percentiles used for discretization. One of "quartile" or "decile"
        """
        self.predict_fn = predict_fn
        self.train_data = train_data

        # define column indices of categorical and ordinal features
        self.categorical_features = sorted(categorical_names.keys())
        self.ordinal_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]

        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names  # dict with {col: categorical feature options}

        # quartile or decile discretization of ordinal features
        args = [self.train_data, self.categorical_features, self.feature_names]
        if discretizer == 'quartile':
            self.disc = lime.lime_tabular.QuartileDiscretizer(*args)
        elif discretizer == 'decile':
            self.disc = lime.lime_tabular.DecileDiscretizer(*args)
        else:
            raise ValueError('Discretizer must be quartile or decile')

        self.d_train_data = self.disc.discretize(self.train_data)

        # add discretized ordinal features to categorical features
        self.categorical_names.update(self.disc.names)
        self.categorical_features += self.ordinal_features

        # calculate min, max and std for ordinal features in training data
        self.min = {}
        self.max = {}
        self.std = {}
        for f in range(self.train_data.shape[1]):
            if f in self.categorical_features and f not in self.ordinal_features:
                continue
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])
            self.std[f] = np.std(train_data[:, f])


    def sample_from_train(self, conditions_eq: dict, conditions_neq: dict,
                          conditions_geq: dict, conditions_leq: dict, num_samples: int):
        """
        Sample data from training set but keep features which are present in the proposed anchor the same
        as the feature value or bin (for ordinal features) as the instance to be explained.

        Parameters
        ----------
        conditions_eq
            Dict: key = feature column; value = categorical feature value
        conditions_neq
            Not used at the moment
        conditions_geq
            Dict: key = feature column; value = bin value of ordinal feature where bin value < feature value
        conditions_leq
            Dict: key = feature column; value = bin value of ordinal feature where bin value >= feature value
        num_samples
            Number of samples used when sampling from training set

        Returns
        -------
        sample
            Sampled data from training set
        """
        train = self.train_data
        d_train = self.d_train_data

        # sample from train and d_train data sets with replacement
        idx = np.random.choice(range(train.shape[0]), num_samples, replace=True)
        sample = train[idx]
        d_sample = d_train[idx]

        # for each sampled instance, use the categorical feature values specified in conditions_eq ...
        # ... which is equal to the feature value in the instance to be explained
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)

        # for the features in condition_geq: make sure sampled feature comes from correct ordinal bin
        for f in conditions_geq:

            # idx of samples where feature value is in a lower bin than the observation to be explained
            idx = d_sample[:, f] <= conditions_geq[f]

            # add idx where feature value is in a higher bin than the observation
            if f in conditions_leq:
                idx = (idx + (d_sample[:, f] > conditions_leq[f])).astype(bool)

            if idx.sum() == 0:
                continue  # if all values in sampled data have same bin as instance to be explained

            # options: idx in train set where with feature value in same bin than instance to be explained
            options = d_train[:, f] > conditions_geq[f]
            if f in conditions_leq:
                options = options * (d_train[:, f] <= conditions_leq[f])

            # if no options, uniformly sample between min and max of feature ...
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:  # ... otherwise draw random samples from training set
                to_rep = np.random.choice(train[options, f], idx.sum(), replace=True)

            # replace sample values for ordinal features where feature values are in a different bin ...
            # ... than the instance to be explained by random values from training set from the correct bin
            sample[idx, f] = to_rep

        # for the features in condition_leq: make sure sampled feature comes from correct ordinal bin
        for f in conditions_leq:

            if f in conditions_geq:
                continue

            idx = d_sample[:, f] > conditions_leq[f]  # idx where feature value is in a higher bin than the observation

            if idx.sum() == 0:
                continue  # if all values in sampled data have same bin as instance to be explained

            # options: idx in train set where with feature value in same bin than instance to be explained
            options = d_train[:, f] <= conditions_leq[f]

            # if no options, uniformly sample between min and max of feature ...
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:  # ... otherwise draw random samples from training set
                to_rep = np.random.choice(train[options, f], idx.sum(), replace=True)
            sample[idx, f] = to_rep

        return sample


    def get_sample_fn(self, X: np.ndarray, desired_label: int = None):
        """
        Create sampling function and mapping dictionary between categorized data and the feature types and values.

        Parameters
        ----------
        X
            Instance to be explained
        desired_label
            Label to use as true label for the instance to be explained

        Returns
        -------
        sample_fn
            Function returning raw and categorized sampled data, and labels
        mapping
            Dict: key = feature column or bin for ordinal features in categorized data; value = tuple containing
                  (feature column, flag for categorical/ordinal feature, feature value or bin value)
        """
        # if no true label available; true label = predicted label
        true_label = desired_label
        if true_label is None:
            true_label = self.predict_fn(X.reshape(1, -1))[0]

        # discretize ordinal features of instance to be explained
        # create mapping = (feature column, flag for categorical/ordinal feature, feature value or bin value)
        mapping = {}
        X = self.disc.discretize(X.reshape(1, -1))[0]
        for f in self.categorical_features:
            if f in self.ordinal_features:
                for v in range(len(self.categorical_names[f])):  # loop over nb of bins for the ordinal features
                    idx = len(mapping)
                    if X[f] <= v and v != len(self.categorical_names[f]) - 1:  # feature value <= bin value
                        mapping[idx] = (f, 'leq', v)  # store bin value
                    elif X[f] > v:  # feature value > bin value
                        mapping[idx] = (f, 'geq', v)  # store bin value
            else:
                idx = len(mapping)
                mapping[idx] = (f, 'eq', X[f])  # store feature value

        def sample_fn(present: list, num_samples: int, compute_labels: bool = True):
            """
            Create sampling function from training data.

            Parameters
            ----------
            present
                List with keys from mapping
            num_samples
                Number of samples used when sampling from training set
            compute_labels
                Boolean whether to use labels coming from model predictions as 'true' labels

            Returns
            -------
            raw_data
                Sampled data from training set
            data
                Sampled data where ordinal features are binned (1 if in bin, 0 otherwise)
            labels
                Create labels using model predictions if compute_labels equals True
            """
            # initialize dicts for 'eq', 'leq', 'geq' tuple value from previous mapping
            # key = feature column; value = feature or bin (for ordinal features) value
            conditions_eq = {}
            conditions_leq = {}
            conditions_geq = {}
            for x in present:
                f, op, v = mapping[x]  # (feature, 'eq'/'leq'/'geq', feature value)
                if op == 'eq':  # categorical feature
                    conditions_eq[f] = v
                if op == 'leq':  # ordinal feature
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)  # store smallest bin > feature value
                if op == 'geq':  # ordinal feature
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)  # store largest bin < feature value

            # sample data from training set
            # feature values are from same discretized bin or category as the explained instance ...
            # ... if defined in conditions dicts
            raw_data = self.sample_from_train(conditions_eq, {}, conditions_geq, conditions_leq, num_samples)

            # discretize sampled data
            d_raw_data = self.disc.discretize(raw_data)

            # use the sampled, discretized raw data to construct a data matrix with the categorical ...
            # ... and binned ordinal data (1 if in bin, 0 otherwise)
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == X[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)

            # create labels using model predictions as true labels
            labels = []
            if compute_labels:
                labels = (self.predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels
        return sample_fn, mapping


    def explain_instance(self, X: np.ndarray, threshold: float = 0.95, delta: float = 0.1,
                         tau: float = 0.15, batch_size: int = 100, max_anchor_size: int = None,
                         desired_label: int = None, **kwargs: dict):
        """
        Explain instance and return anchor with metadata.

        Parameters
        ----------
        X
            Instance to be explained
        threshold
            Minimum precision threshold
        delta
            Used to compute beta
        tau
            Margin between lower confidence bound and minimum precision or upper bound
        batch_size
            Batch size used for sampling
        max_anchor_size
            Maximum number of features in anchor
        desired_label
            Label to use as true label for the instance to be explained

        Returns
        -------
        explanation
            Dictionary containing the anchor explaining the instance with additional metadata
        """
        # build sampling function and ...
        # ... mapping = (feature column, flag for categorical/ordinal feature, feature value or bin value)
        sample_fn, mapping = self.get_sample_fn(X, desired_label=desired_label)

        # get anchors and add metadata
        exp = anchor_base.AnchorBaseBeam.anchor_beam(sample_fn, delta=delta,
                                                     epsilon=tau, batch_size=batch_size,
                                                     desired_confidence=threshold,
                                                     max_anchor_size=max_anchor_size, **kwargs)
        self.add_names_to_exp(exp, mapping)
        exp['instance'] = X
        exp['prediction'] = self.predict_fn(X.reshape(1, -1))[0]
        explanation = anchor_explanation.AnchorExplanation('tabular', exp)
        return explanation


    def add_names_to_exp(self, hoeffding_exp: dict, mapping: dict):
        """
        Add feature names to explanation dictionary.

        Parameters
        ----------
        hoeffding_exp
            Dict with anchors and additional metadata
        mapping
            Dict: key = feature column or bin for ordinal features in categorized data; value = tuple containing
                  (feature column, flag for categorical/ordinal feature, feature value or bin value)
        """
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
